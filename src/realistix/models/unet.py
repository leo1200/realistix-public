"""A small 2D U-Net over the 32x64 grid, with time folded into channels.

RealPDEBench finds a plain U-Net to be the strongest *legal* model on the Foil
scenario -- better than FNO, CNO, MWT, GK-Transformer and Transolver -- and the
competition's starting kit does not ship one. So this is both the obvious first
baseline to beat and, later, the backbone a flow model would denoise with.

Time as channels rather than a third convolution axis: the horizons are fixed at
20 in and 20 out, so there is nothing to gain from time-translation equivariance,
and a 2D network at this resolution is several times cheaper. Inference cost
feeds directly into ``time_score``, where the reference is 0.72896 s/sample and
the budget for a 90+ score is about 9 ms.

The grid is 32x64 and the shipped `--mults 1,2,4` applies **two** downsamplings, so
the bottleneck is **8x16**: `pools` is `Identity` at the last level, because a pool
after the deepest block would feed nothing. (This docstring said "three downsamplings
take it to 4x8" until 2026-08-28. It was wrong for every shipped model.)

Receptive field, measured rather than argued (`tools/stress_checks.py`, check 4): the
half-width along the longest encoder-decoder path is **80 input cells**, so the field
*saturates* a 32x64 grid -- every output cell can see every input cell. Meanwhile 90%
of the gradient mass for one output cell sits within 15-26 cells. The network is
nowhere near reach-limited, which is why every generic long-range-mixing addition has
been flat or worse (attention -0.03, transunet -0.020, FNO lost, XScan null) and why
no further one is worth building.
"""

from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

# Training-only stochastic regularisation (2026-09-10 screen): DropPath on a Block's
# residual branch and channel dropout inside it draw from THIS generator, never from
# the global CUDA RNG, so a noise arm samples exactly the same windows as its control
# (the sampler draws from the global stream). `set_noise_seed` is called once by the
# trainer; with both probabilities at zero nothing here is ever touched.
_NOISE_GEN = None


def set_noise_seed(seed: int, device) -> None:
    global _NOISE_GEN
    _NOISE_GEN = torch.Generator(device=device)
    _NOISE_GEN.manual_seed(int(seed))

T_IN = T_OUT = 20
C = 2  # u, v


def _mod_affine(n_mod: int, ch: int):
    """Per-modality (gain, bias), zero-initialised so it starts as the identity."""
    emb = nn.Embedding(n_mod, 2 * ch)
    nn.init.zeros_(emb.weight)
    return emb


def _apply(emb, h, mod):
    if mod.dtype.is_floating_point:
        # Continuous sim/real fraction (2026-09-10 screen): a float `mod` in [0, 1]
        # interpolates the two modality rows, (1 - a) * row_real + a * row_sim. Exactly
        # the table lookup at a = 0 or 1; long tensors take the shipped path below.
        w = emb.weight
        a = mod.to(w.dtype)[:, None]
        gb = (1 - a) * w[0][None] + a * w[1][None]
    else:
        gb = emb(mod)
    g, b = gb.chunk(2, dim=-1)
    return h * (1 + g[:, :, None, None]) + b[:, :, None, None]


class Block(nn.Module):
    """Two 3x3 convolutions with GroupNorm + SiLU, and a residual connection.

    ``cond_dim`` is the regime-conditional variant: the affine at each normalisation
    site is produced by a linear map from a conditioning vector carrying both the
    modality and a descriptor of the window, instead of being looked up per modality.
    The motivation is measured. Across the 81 paired regimes the simulation tracks
    the real variation well (corr 0.63-0.90 on fluctuation level, wake width, deficit
    depth and cross-flow), but its *levels* are wrong by different factors for
    different quantities -- 3.42x on fluctuations, 2.35x on cross-flow, 1.5x on wake
    width, 1.06x on deficit depth. A per-modality constant cannot express a
    correction that varies by quantity and regime, which is the leading explanation
    for why simulation at an unseen regime hurts rather than helps.

    With ``n_modalities`` the block carries its own per-modality affine at each
    normalisation site. That is the domain-specific-normalisation idea: the
    convolutions stay shared, so simulated windows still teach the real branch its
    dynamics, while the statistics each domain is normalised to are free to differ.
    """

    def __init__(self, cin: int, cout: int, groups: int = 8, n_modalities: int = 0,
                 cond_dim: int = 0, drop_path: float = 0.0, chan_drop: float = 0.0):
        super().__init__()
        # Stochastic regularisation, training only, off by default (bit-identical).
        # drop_path: per-sample Bernoulli on the residual branch after c2, inverted
        # scaling. chan_drop: per-(sample, channel) Bernoulli after the second SiLU,
        # constant over space, inverted scaling. Both read the module-level generator.
        self.drop_path = float(drop_path)
        self.chan_drop = float(chan_drop)
        self.n1 = nn.GroupNorm(min(groups, cin), cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(min(groups, cout), cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.n_modalities = n_modalities
        self.cond_dim = cond_dim
        if n_modalities:
            self.f1 = _mod_affine(n_modalities, cin)
            self.f2 = _mod_affine(n_modalities, cout)
        elif cond_dim:
            self.g1 = nn.Linear(cond_dim, 2 * cin)
            self.g2 = nn.Linear(cond_dim, 2 * cout)
            for lin in (self.g1, self.g2):        # starts as the identity
                nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    @staticmethod
    def _film(lin, h, c):
        g, b = lin(c).chunk(2, dim=-1)
        return h * (1 + g[:, :, None, None]) + b[:, :, None, None]

    def forward(self, x, mod=None):
        h = self.n1(x)
        if self.n_modalities and mod is not None:
            h = _apply(self.f1, h, mod)
        elif self.cond_dim and mod is not None:
            h = self._film(self.g1, h, mod)
        h = self.c1(F.silu(h))
        h = self.n2(h)
        if self.n_modalities and mod is not None:
            h = _apply(self.f2, h, mod)
        elif self.cond_dim and mod is not None:
            h = self._film(self.g2, h, mod)
        h = F.silu(h)
        if self.chan_drop > 0 and self.training:
            keep = (torch.rand(h.shape[0], h.shape[1], 1, 1, device=h.device,
                               generator=_NOISE_GEN) >= self.chan_drop)
            h = h * keep.to(h.dtype) / (1.0 - self.chan_drop)
        h = self.c2(h)
        if self.drop_path > 0 and self.training:
            keep = (torch.rand(h.shape[0], 1, 1, 1, device=h.device,
                               generator=_NOISE_GEN) >= self.drop_path)
            h = h * keep.to(h.dtype) / (1.0 - self.drop_path)
        return h + self.skip(x)


class XScan(nn.Module):
    """Directional exponential scans along the streamwise axis.

    The physics this encodes. The flow is advection-dominated in +x: what happens
    at a downstream cell over the next 20 frames is largely determined by what is
    *upstream* of it now, with influence decaying over a finite correlation length.
    A stack of 3x3 convolutions only expresses that through many attenuating hops;
    this operator states it directly as a first-order recurrence per channel,

        h[x] = a * h[x-1] + (1 - a) * u[x],        a in (0, 1) learned per channel

    -- which is exactly "forward propagation of information with spatial
    correlation decay", and nothing else. A second scan with independent (weaker
    to learn, not constrained) parameters runs upstream, because subsonic flow
    does carry elliptic upstream influence. Executed in convolution form -- the
    kernel (1-a) * a^d materialised per forward -- so it is fully parallel and
    differentiable with none of the aperture-problem gradient pathology that made
    the displacement head (`--warp`) corrupt its anchor.

    Where it can pay and where it cannot. `tools/phase_error.py` rules out an
    advection *error* -- the model already transports at the right speed -- so this
    is not a fix for transport. The claim is about information: the omega in [1,2],
    k < 8 band holds 56% of held-out error at rho ~ 0.65-0.78, its dynamics are
    born in the upstream shear layer, and a receptive field that reaches upstream
    losslessly changes what the network *knows* rather than how it trades -- the
    standing bar (HANDOFF, ConFIG block) for attacking this band a sixth time.
    Generic long-range mixing has failed here three ways (attention -0.03,
    transunet -0.020, FNO lost); the bet is specifically on the directional,
    decay-structured form.

    Zero-initialised output projection, so the module is the exact identity at
    init and a loss is evidence about the mechanism rather than about a harder
    optimisation problem -- the discipline every surviving addition here followed.
    Decay lengths initialise log-spaced from 1 to half the level's width, so the
    channel population starts spanning the plausible correlation lengths.
    """

    def __init__(self, ch: int, max_len: int = 64):
        super().__init__()
        # a = d/(1+d) gives mean memory length d; log-spaced d across channels.
        d = torch.exp(torch.linspace(0.0, math.log(max_len / 2), ch))
        logit = torch.logit(d / (1 + d))
        self.a_fwd = nn.Parameter(logit.clone())
        self.a_bwd = nn.Parameter(logit.clone())
        self.proj = nn.Conv2d(2 * ch, ch, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    @staticmethod
    def _scan(u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Causal depthwise EMA along the last axis, kernel built in fp32."""
        b, c, h, w = u.shape
        d = torch.arange(w, device=u.device, dtype=torch.float32)
        af = a.float().sigmoid().clamp(1e-4, 1 - 1e-4)
        k = (1 - af)[:, None] * af[:, None] ** d[None, :]          # (C, W)
        k = k.flip(-1).reshape(c, 1, w).to(u.dtype)                # conv = corr flip
        v = u.permute(0, 2, 1, 3).reshape(b * h, c, w)
        out = F.conv1d(F.pad(v, (w - 1, 0)), k, groups=c)
        return out.reshape(b, h, c, w).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd = self._scan(x, self.a_fwd)                            # downstream
        bwd = self._scan(x.flip(-1), self.a_bwd).flip(-1)          # upstream
        return x + self.proj(torch.cat([fwd, bwd], dim=1))


class BottleneckAttention(nn.Module):
    """Self-attention over the deepest latent grid.

    Under the shipped `--mults 1,2,4` there are two downsamplings and the latent is
    8x16 = 128 tokens, so global attention costs almost nothing while giving every
    position sight of every other. (Three levels of `mults` give two pools, not
    three -- see the module docstring.) The original motivation was that what shifts
    under a change of angle of attack is the *position* of the wake and a convolution
    stack only sees it through stacked local receptive fields. That motivation is now
    refuted from the other side: the stack's receptive field already saturates the
    domain, which is why this measured -0.03.
    """

    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads,
                                                 h * w).unbind(1)
        a = torch.softmax(q.transpose(-2, -1) @ k / (c // self.heads) ** 0.5, dim=-1)
        o = (v @ a.transpose(-2, -1)).reshape(b, c, h, w)
        return x + self.proj(o)


class TransformerBottleneck(nn.Module):
    """L pre-norm transformer layers over the deepest latent, TransUNet-style.

    ``BottleneckAttention`` is one attention site with no feed-forward and was
    measured at -0.03. This is the stronger form of the same hypothesis -- what a
    TransUNet actually puts at the bottleneck -- so that the family is tested
    rather than a thin proxy for it. The shipped three-level `mults` gives two pools
    and an 8x16 latent, i.e. 128 tokens, so global attention is affordable even
    several layers deep.

    Both output projections are zero-initialised, so the stack starts as the exact
    identity and the run still begins at persistence. That is not cosmetic: it
    makes the variant a strict generalisation of the baseline, so a negative result
    is evidence about the architecture and not about a harder optimisation problem.
    """

    def __init__(self, dim: int, layers: int = 4, heads: int = 8, mlp: int = 4):
        super().__init__()
        self.heads = heads
        self.layers = nn.ModuleList()
        for _ in range(layers):
            attn = nn.ModuleDict(dict(
                norm=nn.GroupNorm(8, dim),
                qkv=nn.Conv2d(dim, dim * 3, 1),
                proj=nn.Conv2d(dim, dim, 1)))
            ff = nn.ModuleDict(dict(
                norm=nn.GroupNorm(8, dim),
                fc1=nn.Conv2d(dim, dim * mlp, 1),
                fc2=nn.Conv2d(dim * mlp, dim, 1)))
            for m in (attn["proj"], ff["fc2"]):
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
            self.layers.append(nn.ModuleList([attn, ff]))

    def forward(self, x):
        b, c, h, w = x.shape
        for attn, ff in self.layers:
            q, k, v = attn["qkv"](attn["norm"](x)).reshape(
                b, 3, self.heads, c // self.heads, h * w).unbind(1)
            a = torch.softmax(q.transpose(-2, -1) @ k / math.sqrt(c // self.heads), dim=-1)
            o = (v @ a.transpose(-2, -1)).reshape(b, c, h, w)
            x = x + attn["proj"](o)
            x = x + ff["fc2"](F.silu(ff["fc1"](ff["norm"](x))))
        return x


class TemporalStem(nn.Module):
    """Shared frame-wise spatial filters, then local temporal and channel mixing.

    The untied control gives each timestamp independent copies of all three
    filters, with the same initial values and RNG consumption as the shared arm.
    Both flatten channel-then-time before the final projection to U-Net width.
    """

    def __init__(self, t_in: int, channels: int, width: int, untied: bool = False):
        super().__init__()
        self.t_in, self.channels, self.untied = t_in, channels, untied
        self.spatial = nn.Conv2d(channels, 8, 3, padding=1)
        self.temporal = nn.Conv3d(8, 8, (3, 1, 1), padding=(1, 0, 0), groups=8)
        self.mix = nn.Conv3d(8, 8, 1)
        self.proj = nn.Conv2d(8 * t_in, width, 1)
        if untied:
            # Copy instead of reinitialising: equal seeds also give equal trunks.
            self.spatial = self._repeat_groups(self.spatial, t_in)
            self.temporal = self._repeat_groups(self.temporal, t_in)
            self.temporal.padding = (0, 0, 0)  # windows are padded before extraction
            self.mix = self._repeat_groups(self.mix, t_in)

    @staticmethod
    def _repeat_groups(conv, copies):
        conv = deepcopy(conv)
        conv.weight = nn.Parameter(conv.weight.repeat(copies, *([1] * (conv.weight.ndim - 1))))
        conv.bias = nn.Parameter(conv.bias.repeat(copies))
        conv.in_channels *= copies
        conv.out_channels *= copies
        conv.groups *= copies
        return conv

    def forward(self, x):
        b, _, h, w = x.shape
        t = self.t_in
        if self.untied:
            z = F.silu(self.spatial(x)).reshape(b, t, 8, h, w)
            z = z.permute(0, 2, 1, 3, 4)  # (B, 8, T, H, W)
            # Each output timestamp owns a depthwise kernel over its three inputs.
            z = F.pad(z, (0, 0, 0, 0, 1, 1)).unfold(2, 3, 1)
            z = z.permute(0, 2, 1, 5, 3, 4).reshape(b, t * 8, 3, h, w)
            z = F.silu(self.mix(self.temporal(z)))
            z = z.reshape(b, t, 8, h, w).permute(0, 2, 1, 3, 4)
        else:
            z = F.silu(self.spatial(x.reshape(b * t, self.channels, h, w)))
            z = z.reshape(b, t, 8, h, w).permute(0, 2, 1, 3, 4)
            z = F.silu(self.mix(self.temporal(z)))
        return self.proj(z.reshape(b, 8 * t, h, w))


class UNet(nn.Module):
    def __init__(self, width: int = 64, mults=(1, 2, 4), blocks_per_level: int = 2,
                 t_in: int = T_IN, t_out: int = T_OUT, channels: int = C,
                 predict_delta: bool = True, n_modalities: int = 0,
                 aux_pressure: bool = False, attn: bool = False,
                 decompose: bool = False, cond: str = "enc",
                 out_gain: bool = False, regime_dim: int = 16,
                 aux_s2r: bool = False, extra_in: int = 0,
                 anchor: str = "last", warp: bool = False, temporal_rank: int = 0,
                 variant: str = "unet", tf_layers: int = 4, xscan: str = "",
                 in_stats: bool = False, mimo: int = 0, out_cumsum: bool = False,
                 stem: str = "dense", geo_in: bool = False,
                 drop_path: float = 0.0, chan_drop: float = 0.0):
        super().__init__()
        # 2026-09-10 regularisation screen. `drop_path` is the branch-drop probability
        # at the widest level; it ramps with log2(channel count) from 0 at the first
        # level (96) to the full value at the widest (384), so 192 gets half.
        # `chan_drop` applies only to blocks at the widest level. Both zero = shipped.
        self._drop_path = float(drop_path)
        self._chan_drop = float(chan_drop)
        if stem not in ("dense", "temporal", "untied"):
            raise ValueError(f"unknown stem: {stem!r}")
        if stem != "dense":
            assert not (in_stats or extra_in or mimo), \
                "temporal/untied stems do not support in_stats, extra_in or mimo"
        # What the predicted delta is added to. The network has always anchored on
        # the *last observed frame*, which is an instantaneous sample of a turbulent
        # field -- i.e. the part of the input that is least predictive of a 0.4 s
        # future. `docs/archive/FINDINGS_2026-09-10.md` §6 localises a third of the residual squared error in
        # the forecast window's 20-frame time-mean, and the input window's own mean
        # is the natural anchor for that component.
        #
        #   last   the shipped anchor: x[:, t_in-1]
        #   wmean  the input window's time-mean, one field for all 20 output frames
        #   blend  a per-output-frame convex mix of the two, learned. This is the
        #          cheapest possible form of the horizon-aware idea: 20 scalars, and
        #          the flow's own physics says the mix should move -- t+1 is nearly
        #          persistence, t+20 is nearly climatology. Initialised at `last`
        #          exactly, so the zero-init output conv still starts training at
        #          persistence and the change is a strict generalisation.
        self.anchor = anchor
        # `--out-cumsum`: the output conv emits per-lead *increments* and the forecast
        # is their cumulative sum over the lead axis on top of the anchor -- every output
        # channel has the magnitude of a one-step change and the structure is a
        # time-stepper's, without feeding predictions back in. A linear
        # reparameterisation of the same function class; zero-init still starts at
        # persistence. Screened on the mimic folds 2026-09-05.
        self.out_cumsum = out_cumsum
        # Topology within the U-Net family. The operator families (FNO, CNO,
        # Transolver, DPOT) are closed by RealPDEBench's own evaluation on this
        # flow, where a plain U-Net at 23.0M wins outright -- but that benchmark
        # says nothing about *which* U-Net, and width, depth, blocks and level
        # count are the only axes swept here so far.
        #
        #   unet      concatenating skips, the shipped topology
        #   linknet   additive skips. The encoder feature is added to the upsampled
        #             decoder feature rather than concatenated, which halves the
        #             first decoder block's input width and forces the two to share
        #             a representation instead of being mixed by a convolution
        #   nested    UNet++-style dense skips, in their cheap form: each skip is
        #             re-processed by its own block before the decoder sees it, so
        #             the encoder feature crossing the gap is not raw
        #   transunet a multi-layer transformer at the bottleneck
        #
        # All three start at or near the baseline by construction, so a loss is
        # evidence about topology rather than about optimisation difficulty.
        self.variant = variant
        # Predict a displacement field and *transport* the anchor, instead of asking
        # the convolution stack to synthesise advected structure from scratch. The
        # flow is advection-dominated and this is the last physical prior available
        # (`docs/archive/ideas_2026-09-10.md` item 3): divergence-free is ruled out by measurement and
        # pressure by argument. Zero-initialised, so zero displacement -- the model
        # again starts exactly at persistence and warping is something it learns.
        self.warp = warp
        # Half the field width. The displacement head is bounded by this through a
        # tanh, so a sample can leave the frame -- which is honest, structure really
        # does advect out -- but the field cannot run away to infinity.
        self.max_disp = 32.0
        # Where the modality is allowed to act. The original design conditioned the
        # encoder only, which is the half of the network that reads statistics
        # rather than the half that produces them -- and the sim/real gap is a
        # *fluctuation-energy* gap (sim carries a median 3.8x the energy after
        # normalisation), i.e. a property of the output.
        #
        #   enc   after each encoder level                     (the original)
        #   all   encoder, bottleneck and decoder
        #   norm  a per-modality affine at every normalisation site, inside the
        #         blocks; convolutions stay shared
        #   film_re  the same sites, but the affine is a linear function of
        #         [modality, TRUE (Re, AoA)]. The distinction from `norm_re` is the
        #         whole point: there the regime descriptor was read off the window,
        #         so a simulated trajectory and its real partner -- which differ by
        #         3.4x in fluctuation energy -- produced different codes, and nothing
        #         told the network they were the same flow. Here the code is identical
        #         by construction and only the modality token differs, so learning
        #         about a regime from simulation updates parameters that also serve
        #         real predictions there. That coupling is what regime transfer needs
        #         and what training otherwise never supplies.
        #   norm_re  the same sites, but the affine is a linear function of
        #         [modality, regime], where the regime descriptor is read off the
        #         window itself -- the velocity scale carries Reynolds number and the
        #         zero-mask geometry carries angle of attack. Strictly generalises
        #         `norm`, and zero-initialised so it starts there
        #
        # ``out_gain`` adds a per-modality scale on the predicted delta, which is
        # the most direct parameterisation of an energy mismatch there is.
        self.cond = cond if n_modalities else "none"
        self.out_gain = out_gain and bool(n_modalities)
        self.t_in, self.t_out, self.c = t_in, t_out, channels
        # T1 (2026-09-17): a fixed low-rank temporal output basis. With temporal_rank R > 0 the output
        # convolution emits R coefficient fields per channel and the 20 frames are decoded through the
        # first R orthonormal DCT-II modes (analytic, not learned). 0 = the shipped 20-frame output,
        # bit-identical. Feasibility: the R = 12 projection floor is 13-15 % of the ensemble's rel error
        # (CANDIDATE_t1.md), so the restriction tests generalisation, not capacity.
        self.temporal_rank = int(temporal_rank or 0)
        if self.temporal_rank:
            if mimo or decompose or warp or anchor != "last":
                raise ValueError("temporal_rank composes with the plain last-anchor output only")
            n = torch.arange(t_out, dtype=torch.float32)
            D = torch.stack([torch.cos(math.pi * (n + 0.5) * k / t_out) for k in range(self.temporal_rank)]) * math.sqrt(2.0 / t_out)
            D[0] /= math.sqrt(2.0)
            self.register_buffer("dct", D.t().contiguous())      # (t_out, R)
        self.predict_delta = predict_delta
        # FiLM on modality. Training jointly on real and simulated windows without
        # telling the model which is which forces it to average two regimes that
        # differ by several times in fluctuation energy. With a modality embedding
        # it can represent the gap instead, and inference always passes the real token.
        self.n_modalities = n_modalities
        # Pressure is unmeasured in the real data but present in simulation. As an
        # auxiliary target on simulated windows only, it forces the trunk to carry
        # the velocity-pressure coupling, which is physics that should transfer.
        self.aux_pressure = aux_pressure
        # MVPE scores the 20-frame time-mean at 36 probes, and under angle shift it
        # is our weakest quantity. With `decompose` the network predicts that mean
        # through its own head and the fluctuations about it separately, so the
        # quantity MVPE measures has a dedicated pathway instead of emerging as a
        # by-product of instantaneous prediction. The fluctuation branch is forced
        # to average to zero, which makes the split exact.
        self.decompose = decompose
        # ``extra_in`` channels are appended *after* the input window, so the
        # persistence residual below still slices the last observed frame at
        # (t_in-1)*c. Used only by tools/info_limit.py, which feeds the network part
        # of its own target to bound the reachable error; it is 0 on every shipped
        # path and build_model never sets it.
        # Window statistics as explicit input channels (2026-08-29). The anatomy
        # measured that ~half the emitted correction is driven by the input
        # window's 20-frame mean and that the oldest ten frames carry 2.5% of the
        # occlusion effect -- so the stem's input capacity is mostly unread, and
        # the two summary fields the network provably uses (the time-mean) or is
        # architecturally bad at computing (the per-cell temporal std, which needs
        # squaring) can be handed over for 4 extra channels at a stem that is
        # 0.035M of 22.65M. Computed inside forward from x itself, so every call
        # site -- training, eval, the head's ensemble, a packed submission --
        # works unchanged.
        self.in_stats = in_stats
        # MIMO (2026-08-31): M output groups hosted in one trunk and one forward
        # pass. Training feeds M independently sampled windows concatenated in
        # channels, and each group's loss is its own window's future -- so the
        # groups are M implicit ensemble members sharing capacity. At inference
        # every call site still hands a single 40-channel window; forward tiles it
        # M times, returns the GROUP MEAN (drop-in compatible with eval, the head
        # and the packed submission), and stashes the per-group stack on
        # ``self.mimo_members`` for a spread reader. The capacity this spends is
        # the width that is measured dead for a single model (128/160/192 within
        # seed noise of 96), and the prize is the ensemble's 2 x 0.85 ms.
        self.mimo = int(mimo) if mimo and mimo > 1 else 0
        if self.mimo and (decompose or warp or out_gain or aux_pressure or aux_s2r
                          or anchor != "last" or extra_in):
            raise ValueError("mimo composes only with the plain delta path; "
                             "measure it alone first")
        m = self.mimo or 1
        # ``geo_in`` (bold round, 2026-09-06): two channels computed inside forward from the
        # window's own exact-zero mask -- the invalid mask (foil body + laser shadow) and a
        # quantised distance-to-nearest-invalid-cell field (12 dilations of a 3x3 max-pool,
        # scaled to [0, 1]). The network already sees the zeros; this hands it the smooth
        # geometric variable that changes continuously with angle of attack, so angle can
        # be read as geometry rather than inferred from wake position. Every call site works
        # unchanged, exactly as with ``in_stats``; off = bit-identical.
        self.geo_in = bool(geo_in)
        if self.geo_in and (mimo or stem != "dense"):
            raise ValueError("geo_in composes with the dense stem only")
        cin = m * t_in * channels + extra_in + (2 * channels if in_stats else 0) + (2 if geo_in else 0)
        cout = m * (int(temporal_rank or 0) or t_out) * channels

        # int() rather than plain multiplication, because `mults` may now carry
        # fractions: every width sweep in this project scaled all levels together
        # (64 -> 96 -> 128 -> 160 -> 192) and the untried axis is the *profile* --
        # holding the parameter budget and moving it toward the shallow levels,
        # where the ablation says the function lives (docs/archive/model_analysis.md 2.4).
        # Integer mults give exactly the previous channel counts, so no checkpoint
        # changes meaning.
        # Rounded to a multiple of 8, because `mults` may now carry fractions and
        # every Block's GroupNorm asks for 8 groups. Integer mults at any width
        # already used here land on multiples of 8 unchanged, so no checkpoint
        # changes meaning.
        #
        # Why fractions at all: every width sweep in this project scaled all three
        # levels together (64 -> 96 -> 128 -> 160 -> 192). The untried axis is the
        # *profile* -- moving the budget toward the shallow levels, where the
        # ablation says the function lives (docs/archive/model_analysis.md 2.4: the deepest
        # skip costs 0.009 of mean4 to remove and the bottleneck 0.267). Match on
        # FLOPs rather than on parameters: cost is dominated by the full-resolution
        # level, so a parameter-matched reallocation is a 1.7x slower network, and
        # `time_score` carries 0.095 of the objective (the "+0.147 hard ceiling" this
        # comment cited was retracted 2026-09-03 -- FINDINGS 9c).
        chans = [max(8, int(round(width * m / 8)) * 8) for m in mults]
        c_lo, c_hi = chans[0], chans[-1]

        def _noise(ch):
            ramp = (math.log2(ch / c_lo) / math.log2(c_hi / c_lo)) if c_hi > c_lo else 1.0
            return dict(drop_path=self._drop_path * max(0.0, min(1.0, ramp)),
                        chan_drop=self._chan_drop if ch == c_hi else 0.0)

        self.stem = (nn.Conv2d(cin, chans[0], 3, padding=1) if stem == "dense" else
                     TemporalStem(t_in, channels, chans[0], untied=stem == "untied"))

        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = chans[0]
        skip_chans = [prev]
        for i, ch in enumerate(chans):
            nb, cd = self._block_cond(n_modalities, regime_dim)
            level = nn.ModuleList([Block(prev if b == 0 else ch, ch, n_modalities=nb,
                                         cond_dim=cd, **_noise(ch))
                                   for b in range(blocks_per_level)])
            self.downs.append(level)
            prev = ch
            skip_chans.append(prev)
            self.pools.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                              if i < len(chans) - 1 else nn.Identity())

        nb, cd = self._block_cond(n_modalities, regime_dim)
        self.mid1 = Block(prev, prev, n_modalities=nb, cond_dim=cd, **_noise(prev))
        self.mid2 = Block(prev, prev, n_modalities=nb, cond_dim=cd, **_noise(prev))

        # Directional streamwise scans (see XScan). "deep" places them where the
        # meandering-scale content lives (the last encoder level and the
        # bottleneck); "all" adds every encoder level. Feature width at level i is
        # 64 / 2^i, and the scan spans the whole of it either way.
        self.xscan = xscan
        self.xscan_mods = nn.ModuleDict()
        if xscan:
            lo = len(chans) - 1 if xscan == "deep" else 0
            for i in range(lo, len(chans)):
                self.xscan_mods[f"enc{i}"] = XScan(chans[i], max_len=max(4, 64 >> i))
            self.xscan_mods["mid"] = XScan(prev, max_len=max(4, 64 >> len(chans)))

        self.ups = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, ch in reversed(list(enumerate(chans))):
            self.upsamples.append(nn.ConvTranspose2d(prev, ch, 4, stride=2, padding=1)
                                  if i < len(chans) - 1 else nn.Identity())
            prev = ch if i < len(chans) - 1 else prev
            # linknet adds the skip instead of concatenating it, so the first
            # decoder block sees `ch` rather than `2*ch`. After `up` the feature and
            # its skip already carry the same channel count at every level, so the
            # addition needs no projection.
            first_in = prev if variant == "linknet" else prev + ch
            level = nn.ModuleList([Block(first_in if b == 0 else ch, ch,
                                         n_modalities=nb, cond_dim=cd, **_noise(ch))
                                   for b in range(blocks_per_level)])
            self.ups.append(level)
            prev = ch

        # `--attn` is the one-site version, measured at -0.03; `transunet` is the
        # multi-layer form the family is actually named for. They occupy the same
        # slot, so asking for both is a configuration error rather than a stack.
        if variant == "transunet":
            if attn:
                raise ValueError("--attn and --variant transunet both occupy the "
                                 "bottleneck; pick one")
            self.attn = TransformerBottleneck(chans[-1], layers=tf_layers)
        else:
            self.attn = BottleneckAttention(chans[-1]) if attn else None

        # UNet++ in its cheap form: one block per skip, so what crosses the gap has
        # been processed rather than lifted straight out of the encoder. Zero-cost
        # to reason about because the block is residual -- it starts near identity.
        self.skip_blocks = None
        if variant == "nested":
            nb, cd = self._block_cond(n_modalities, regime_dim)
            self.skip_blocks = nn.ModuleList(
                [Block(ch, ch, n_modalities=nb, cond_dim=cd) for ch in chans])

        self.out_norm = nn.GroupNorm(8, prev)
        self.out = nn.Conv2d(prev, cout, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        if decompose:
            self.out_mean = nn.Conv2d(prev, channels, 3, padding=1)
            nn.init.zeros_(self.out_mean.weight); nn.init.zeros_(self.out_mean.bias)
        if anchor == "blend":
            # sigmoid(4) = 0.982, so the mix starts on the last frame and the run
            # still begins at persistence to three decimals. A logit rather than a
            # clamped weight so the optimiser sees no boundary.
            self.anchor_logit = nn.Parameter(torch.full((t_out,), 4.0))
        if warp:
            self.warp_head = nn.Conv2d(prev, t_out * 2, 3, padding=1)
            nn.init.zeros_(self.warp_head.weight); nn.init.zeros_(self.warp_head.bias)

        # An auxiliary head that emits a correction to the *input* window, trained to
        # make a simulated window's statistics match real measurements at the same
        # regime. No synthetic sample is ever materialised -- the training set stays
        # the release -- but the trunk is forced to represent the sim->real
        # relationship, which is the thing four conditioning schemes failed to make
        # usable. The hypothesis is that this helps precisely where real data is
        # absent and only simulation covers the regime.
        self.aux_s2r = aux_s2r
        if aux_s2r:
            self.s2r_head = nn.Conv2d(prev, t_in * channels, 3, padding=1)
            nn.init.zeros_(self.s2r_head.weight); nn.init.zeros_(self.s2r_head.bias)

        if aux_pressure:
            self.p_head = nn.Conv2d(prev, t_out, 3, padding=1)
            nn.init.zeros_(self.p_head.weight); nn.init.zeros_(self.p_head.bias)

        # `norm` conditions inside the blocks, so the level-wise FiLM path is not
        # built at all -- otherwise the checkpoint carries half a megabyte of
        # parameters that no forward pass ever reads.
        if n_modalities and self.cond in ("enc", "all"):
            self.mod_emb = nn.Embedding(n_modalities, width * 4)
            sites = list(chans)
            if self.cond == "all":
                sites += [chans[-1]] + list(reversed(chans))   # bottleneck, decoder
            self.film = nn.ModuleList([nn.Linear(width * 4, 2 * ch) for ch in sites])
            for lin in self.film:            # start as the identity transform
                nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
            self.n_enc_sites = len(chans)
        if self.cond in ("norm_re", "film_re"):
            # A descriptor of *which flow this is*, read off the window: the encoder
            # sees the zero-mask geometry (angle of attack, recoverable to 98.8%) and
            # log_scale carries the Reynolds number (the velocity scale that per-window
            # normalisation divides out, recoverable to 3.6%).
            if self.cond == "norm_re":
                self.regime = nn.Sequential(
                    nn.Conv2d(cin, 32, 3, stride=2, padding=1), nn.SiLU(),
                    nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU(),
                    nn.AdaptiveAvgPool2d(1), nn.Flatten())
                self.regime_mlp = nn.Sequential(nn.Linear(33, 64), nn.SiLU(),
                                                nn.Linear(64, regime_dim))
            else:
                # (log Re, sin/cos of the angle) -- smooth in both parameters, so a
                # regime never seen in training still lands somewhere sensible.
                self.regime_mlp = nn.Sequential(nn.Linear(3, 64), nn.SiLU(),
                                                nn.Linear(64, regime_dim))
            self.mod_vec = nn.Embedding(n_modalities, 8)
            nn.init.zeros_(self.mod_vec.weight)
        if n_modalities:
            if self.out_gain:
                self.delta_gain = nn.Embedding(n_modalities, 1)
                nn.init.zeros_(self.delta_gain.weight)

    def _block_cond(self, n_modalities, regime_dim):
        """(embedding-based modalities, linear-conditioning width) for the blocks."""
        if self.cond == "norm":
            return n_modalities, 0
        if self.cond in ("norm_re", "film_re"):
            return 0, 8 + regime_dim          # modality vector, then regime descriptor
        return 0, 0

    def translate(self, x, modality=None, log_scale=None, regime=None):
        """Simulated window -> real-looking window, through the forecaster's trunk."""
        return x + self.forward(x, modality, log_scale, regime, want=1)

    def forward(self, x, modality=None, log_scale=None, regime=None, want=0):
        """(B, T_in*C, H, W) -> (B, T_out*C, H, W), both in normalized units.

        ``modality`` is a (B,) long tensor of dataset ids (0 = real) when the model
        was built with ``n_modalities``; None falls back to real.
        """
        e = None
        if self.n_modalities:
            if modality is None:
                modality = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            if self.cond in ("enc", "all"):
                e = self.mod_emb(modality)
        cvec = None
        if self.cond == "norm_re":
            # Descriptor inferred from the window: the two modalities produce
            # different codes for the same flow, which is why this did not work.
            ls = (log_scale if log_scale is not None
                  else torch.zeros(x.shape[0], device=x.device, dtype=x.dtype))
            z = self.regime_mlp(torch.cat([self.regime(x), ls.reshape(-1, 1)], dim=1))
            cvec = torch.cat([self.mod_vec(modality), z], dim=1)
        elif self.cond == "film_re":
            # Descriptor from the true labels: identical for a simulated trajectory
            # and its real partner, so the regime pathway is shared between them.
            if regime is None:
                regime = x.new_zeros(x.shape[0], 2)
            re = regime[:, 0].clamp_min(1.0)
            aoa = regime[:, 1] * (math.pi / 180.0)
            feat = torch.stack([torch.log(re) - 9.5, torch.sin(aoa), torch.cos(aoa)], 1)
            cvec = torch.cat([self.mod_vec(modality), self.regime_mlp(feat)], dim=1)
        # `norm` blocks look the modality up; `norm_re` blocks take the joint vector.
        m = modality if self.cond == "norm" else cvec
        deep = self.cond == "all"                          # bottleneck and decoder too

        def film(h, k):
            g, b = self.film[k](e).chunk(2, dim=-1)
            return h * (1 + g[:, :, None, None]) + b[:, :, None, None]

        mimo_tiled = False
        if self.mimo:
            if x.shape[1] == self.t_in * self.c:      # a single window: inference
                x = x.repeat(1, self.mimo, 1, 1)
                mimo_tiled = True
            assert x.shape[1] == self.mimo * self.t_in * self.c, \
                "mimo model fed neither one window nor M windows"
        if self.in_stats:
            xs = x[:, :self.t_in * self.c].reshape(
                x.shape[0], self.t_in, self.c, *x.shape[-2:])
            stats = torch.cat([xs.mean(dim=1), xs.std(dim=1)], dim=1)
            h = self.stem(torch.cat([x, stats], dim=1))
        elif self.geo_in:
            with torch.no_grad():
                invalid = (x[:, :self.t_in * self.c].abs() < 1e-8).all(dim=1, keepdim=True)  # (B,1,H,W)
                inv = invalid.to(x.dtype)
                dist = torch.zeros_like(inv)
                cover = inv
                for _ in range(12):
                    cover = F.max_pool2d(cover, 3, stride=1, padding=1)
                    dist = dist + (1.0 - cover)          # counts dilations a cell stays uncovered
                dist = dist / 12.0
            h = self.stem(torch.cat([x, inv, dist], dim=1))
        else:
            h = self.stem(x)
        skips = []
        for i, (level, pool) in enumerate(zip(self.downs, self.pools)):
            for blk in level:
                h = blk(h, m)
            if e is not None and self.cond in ("enc", "all"):
                h = film(h, i)
            if f"enc{i}" in self.xscan_mods:
                h = self.xscan_mods[f"enc{i}"](h)
            skips.append(h)
            h = pool(h)

        h = self.mid2(self.mid1(h, m), m)
        if "mid" in self.xscan_mods:
            h = self.xscan_mods["mid"](h)
        if e is not None and deep:
            h = film(h, self.n_enc_sites)
        if self.attn is not None:
            h = self.attn(h)

        if self.skip_blocks is not None:
            skips = [blk(sk, m) for blk, sk in zip(self.skip_blocks, skips)]

        for j, (level, up, skip) in enumerate(
                zip(self.ups, self.upsamples, reversed(skips))):
            h = up(h)
            # Additive skips carry no extra channels, so LinkNet's decoder is
            # narrower at its first block and the two features must agree on a
            # representation rather than being reconciled by a convolution.
            h = h + skip if self.variant == "linknet" else torch.cat([h, skip], dim=1)
            for blk in level:
                h = blk(h, m)
            if e is not None and deep:
                h = film(h, self.n_enc_sites + 1 + j)

        feat = F.silu(self.out_norm(h))
        if want == 2:
            # The decoder's final feature map, tapped for the interval head. It is
            # computed by every member pass we already pay for, so reading it costs
            # no inference time -- and unlike the members' *output* disagreement it
            # reports where the networks internally disagree about what structure is
            # present, even where their emitted fields happen to coincide. That is
            # the only uncertainty signal available that is not a function of
            # (window, prediction, output-spread), which tools/k_anatomy.py showed
            # the head already saturates (2026-08-28).
            return feat
        if want == 1:                                   # the sim->real correction
            return self.s2r_head(feat)
        y = self.out(feat)
        if self.temporal_rank:
            b_, _, hh_, ww_ = y.shape
            coef = y.reshape(b_, self.temporal_rank, self.c, hh_, ww_)
            y = torch.einsum("tr,brchw->btchw", self.dct.to(y.dtype), coef).reshape(b_, self.t_out * self.c, hh_, ww_)
        p_out = self.p_head(feat) if self.aux_pressure else None
        if self.mimo:
            b, _, hh, ww = x.shape
            last = x.reshape(b, self.mimo, self.t_in, self.c, hh, ww)[:, :, -1]
            last = last.reshape(b, self.mimo * self.c, hh, ww)
        else:
            last = x[:, (self.t_in - 1) * self.c: self.t_in * self.c]

        if self.predict_delta and not self.decompose:
            # (B, t_out*c, H, W): one anchor field per output frame, so `blend` and
            # `warp` can both vary with lead time. `last` alone is the shipped case
            # and is still exactly `last.repeat(1, t_out, 1, 1)`.
            if self.mimo:
                # Group-major layout throughout: group g's t_out frames anchor on
                # group g's own last frame, so every group starts at persistence.
                b, _, hh, ww = x.shape
                base = (last.reshape(b, self.mimo, 1, self.c, hh, ww)
                        .expand(b, self.mimo, self.t_out, self.c, hh, ww)
                        .reshape(b, self.mimo * self.t_out * self.c, hh, ww))
            else:
                base = self._anchor(x, last)
            if self.warp:
                base = self._transport(base, self.warp_head(feat))

        if self.decompose:
            # Fluctuations, made exactly zero-mean in time so the mean head alone
            # determines the time-average -- the quantity MVPE scores.
            f = y.reshape(-1, self.t_out, self.c, *y.shape[-2:])
            f = f - f.mean(dim=1, keepdim=True)
            # What the mean head's zero-init correction is added to. `last` is the
            # original decompose (measured flat, pooled, 2026-08-17). `wmean` is the
            # 2026-08-21 edge-bias candidate: the input window's own time-mean
            # carries the *true* climatology at any regime, including ones past the
            # training range, so the default output mean under extrapolation is the
            # observed one rather than a mean synthesised by a network that was
            # never constrained out there. The measured target: the constant probe
            # offset that is 29%/22% of squared error at held-out edge cells
            # against 6%/5% in-sample (HANDOFF, modelling session of 2026-08-21).
            if self.predict_delta:
                if self.anchor == "last":
                    base_m = last
                else:
                    base_m = x[:, :self.t_in * self.c].reshape(
                        -1, self.t_in, self.c, *x.shape[-2:]).mean(dim=1)
            else:
                base_m = 0.0
            corr = self.out_mean(feat)
            # Stashed for the optional `--w-mcorr` penalty: the correction is a
            # learned field and therefore extrapolates with the same bias the
            # anchor exists to remove, so the training loop may tax its magnitude
            # to push the mean's out-of-distribution default onto the anchor.
            self.mcorr = corr
            mean = corr + base_m
            y = (mean[:, None] + f).reshape(y.shape)
        elif self.predict_delta:
            # Anchor on the last observed frame. The output conv is zero-initialized,
            # so training starts exactly at persistence and moves away from it --
            # which also means the TKE floor of 66.67 is the starting point, not a
            # trap: the delta is what has to learn the fluctuations.
            if self.out_gain:
                # One scalar per modality on the delta. The delta *is* the predicted
                # fluctuation, and the domain gap is that simulation carries several
                # times the fluctuation energy, so this is the smallest parameterisation
                # that can express the gap at all.
                y = y * (1 + self.delta_gain(modality)[:, :, None, None])
            if self.out_cumsum:
                assert not self.mimo, "--out-cumsum is not implemented for MIMO groups"
                b, _, hh, ww = y.shape
                y = y.reshape(b, self.t_out, self.c, hh, ww).cumsum(dim=1).reshape(b, -1, hh, ww)
            y = y + base
        if self.mimo and mimo_tiled:
            b, _, hh, ww = y.shape
            stack = y.reshape(b, self.mimo, self.t_out * self.c, hh, ww)
            # The per-group forecasts, for whoever wants a spread; the returned
            # prediction is their mean, so every existing call site is unchanged.
            self.mimo_members = stack
            y = stack.mean(dim=1)
        return (y, p_out) if self.aux_pressure else y

    def _anchor(self, x, last):
        """The field each output frame's delta is added to, (B, t_out*c, H, W)."""
        if self.anchor == "last":
            return last.repeat(1, self.t_out, 1, 1)
        b, _, h, w = x.shape
        wmean = x[:, :self.t_in * self.c].reshape(
            b, self.t_in, self.c, h, w).mean(dim=1)
        if self.anchor == "wmean":
            return wmean.repeat(1, self.t_out, 1, 1)
        a = torch.sigmoid(self.anchor_logit).to(x.dtype)[None, :, None, None, None]
        mix = a * last[:, None] + (1 - a) * wmean[:, None]        # (B, t_out, c, H, W)
        return mix.reshape(b, self.t_out * self.c, h, w)

    def _transport(self, base, disp):
        """Advect each anchor frame by its own predicted displacement, in cells.

        `disp` is zero-initialised, and grid_sample at zero displacement is the
        identity, so this starts as a no-op and the residual head is unaffected. The
        border padding matters: at 0.4 s the true displacement of a wake structure
        can exceed the 64-cell field, and clamping is the honest behaviour there --
        the information has simply left the frame.
        """
        b, _, h, w = base.shape
        t = self.t_out
        # Two things this line carries, both learned the hard way.
        #
        # The displacement is bounded through a tanh rather than emitted raw. An
        # unbounded field is a bad parameterisation on its own terms -- border
        # padding makes the gradient vanish once a sample leaves the frame, so the
        # head is free to drift arbitrarily far with nothing pulling it back -- and
        # under fp16 that drift overflows. It did: `--warp` trained fine for 400
        # steps and then produced non-finite outputs somewhere before step 1000,
        # surfacing as the scorer's "SPS bounds must be finite". Half the field
        # width is far more displacement than 0.4 s of this flow can produce.
        #
        # And the sampling itself runs in fp32. grid_sample interpolates
        # coordinates, so it is exactly the operation where half precision buys
        # nothing and costs conditioning; the cast is free at 32x64.
        d = self.max_disp * torch.tanh(disp.reshape(b * t, 2, h, w).float())
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=base.device, dtype=torch.float32),
            torch.linspace(-1, 1, w, device=base.device, dtype=torch.float32),
            indexing="ij")
        gx = xx[None] + d[:, 0] * (2.0 / max(w - 1, 1))
        gy = yy[None] + d[:, 1] * (2.0 / max(h - 1, 1))
        grid = torch.stack([gx, gy], dim=-1)
        out = F.grid_sample(base.reshape(b * t, self.c, h, w).float(), grid,
                            mode="bilinear", padding_mode="border",
                            align_corners=True)
        return out.reshape(b, t * self.c, h, w).to(base.dtype)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
