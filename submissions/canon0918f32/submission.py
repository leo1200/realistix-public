"""Track 1 submission: 3-seed U-Net ensemble + learned asymmetric SPS intervals.

Self-contained: torch and numpy only, no vendored kit code. Trained by
``tools/train_final.sh`` on the 81 usable real trajectories and all 100 simulated
ones of the official release, and on nothing else. (The release ships 82 real files;
``7575_0`` was withdrawn by the organisers on 2026-08-12 because it reuses
``6300_0``'s raw measurements, and ``splits.EXCLUDED`` drops it.)

Four things this does that a plain wrapped baseline does not.

Per-window normalization. Real velocities are in m/s and scale with Reynolds
number across a factor of six, while simulated ones are non-dimensional; each
window is divided by its own RMS speed over valid cells. Re and AoA are not
available at inference (``metadata`` is an empty dict on scored calls), but both
are recoverable from the window itself -- AoA to 98.8% from the zero-mask
geometry, Re to 3.6% from the velocity scale -- so nothing is lost by not being
told them.

Learned asymmetric intervals. SPS pays ``exp(-(upper-lower)/sigma)`` on covered
elements and nothing on uncovered ones, so the objective is steeply asymmetric: a
2x-too-wide interval costs 20-30% of its reward -- ``1 - exp(-width/sigma)`` at the
0.22-0.35 sigma widths this head settles on -- while missing coverage costs all of it. Setting the width from a predicted error is therefore mis-specified -- per-cell
means, per-cell quantiles and input-variability proxies all score *below* a tuned
constant band. The head instead outputs both bounds and is trained on the reward
itself. It also need not centre the interval on the prediction, which matters
because SPS charges for total width: an interval reaching from prediction to
target costs half a symmetric one.

Ensemble spread as an input to those intervals. The three members disagree, and
where they disagree the ensemble mean is more likely to be wrong; that is
instance-specific uncertainty produced by forward passes we are making anyway.
The head reads the per-element standard deviation across members, in the same
units as sigma_global. (It was worth +0.4 SPS when introduced; re-measured
2026-08-31 against current members the input is worth ~0 -- the head reconstructs
what it needs from (window, prediction). Kept because removal is not a gain and
this build is the verified one. docs/MODEL.md carries the ledger.)

CUDA-graph replay at the container's batch size. Submission 912717 moved setup out
of the timed region and read 6.65 ms/sample where the same code runs at ~1.1 ms/sample
on a large batch and ~9-18 ms per *call* on one to eight windows -- i.e. the container
calls predict() on a few windows at a time and the clock is kernel-launch latency,
not arithmetic (docs: FINDINGS section 9c). Each bucketed batch shape is captured once
at import as a CUDA graph -- the three members on three streams, then the head -- and
predict() replays it. Outputs match the eager path to one fp16 ulp; if capture or a
self-check fails the eager path below is used unchanged.

fp16 output. After the GPU work, copying three full-size arrays back to the host
was 69% of inference wall time. Halving the bytes cuts that to a quarter, at a
cost of 3e-6 in relative L2 -- five orders below the error being measured.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T_IN = T_OUT = 20
C = 2
EPS = 1e-8

_HERE = os.path.dirname(os.path.abspath(__file__))
_CKPT = os.path.join(_HERE, "model.pth")
#: k4g (2026-09-12): member-variance-matched fluctuation gain on the ensemble mean. Averaging
#: K members damps each cell's temporal fluctuation by the members' disagreement; the
#: scorer's TKE pays for that damping. Per window and cell the mean's temporal variance is
#: rescaled toward the members' own average temporal variance -- a quantity the members
#: supply, no target involved -- at dose GAIN_DOSE (0 = the plain mean, 1 = full match).
#: The time-mean of every cell is unchanged, so MVPE is untouched; the interval head still
#: reads and brackets the PLAIN mean, so the intervals are k4's (the lo<=pred<=hi guard
#: below may widen a few elements where the gained prediction leaves them).
GAIN_DOSE = 0.5
GAIN_CAP = 3.0


def _matched_gain(outs, y):
    """outs (K, B, T*C, H, W) members, y (B, T*C, H, W) their mean, both in normalised units.
    Returns y with each (window, channel, cell) fluctuation about its time-mean scaled by
    1 + dose * (sqrt(v_members / v_mean) - 1), capped."""
    k, b, tc, h, w = outs.shape
    o = outs.reshape(k, b, T_OUT, C, h, w).float()
    ym = y.reshape(b, T_OUT, C, h, w).float()
    mu = ym.mean(dim=1, keepdim=True)
    v_mean = ((ym - mu) ** 2).mean(dim=1, keepdim=True)
    v_mem = ((o - o.mean(dim=2, keepdim=True)) ** 2).mean(dim=2, keepdim=True).mean(dim=0)
    g = torch.sqrt(v_mem / v_mean.clamp_min(1e-12)).clamp(max=GAIN_CAP)
    g = 1.0 + GAIN_DOSE * (g - 1.0)
    return (mu + g * (ym - mu)).reshape(b, tc, h, w).to(y.dtype)
_BATCH = 256


class Block(nn.Module):
    """With ``n_modalities`` the block carries a per-domain affine at each norm.

    The convolutions stay shared, so simulated windows still teach the real branch
    its dynamics, while the statistics each domain is normalised to may differ.
    (Worth +0.17 under the 2026-08 MSE-era loss; re-measured 2026-08-31 under the
    final recipe the conditioning's marginal is inside noise -- the modality is
    perfectly separable from the input via the optical shadow, and the rel-
    dominated loss removed the averaging conflict. Kept: folds away at export.)
    The original placement argument, retained for the record:
    the sim/real gap is a fluctuation-energy gap, and energy is set where the
    network produces output, not where it reads input. Inference always selects
    the real row.
    """

    def __init__(self, cin, cout, groups=8, n_modalities=0):
        super().__init__()
        self.n1 = nn.GroupNorm(min(groups, cin), cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(min(groups, cout), cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.n_modalities = n_modalities
        if n_modalities:
            self.f1 = nn.Embedding(n_modalities, 2 * cin)
            self.f2 = nn.Embedding(n_modalities, 2 * cout)

    def _mod(self, emb, h, mod):
        g, b = emb(mod).chunk(2, dim=-1)
        return h * (1 + g[:, :, None, None]) + b[:, :, None, None]

    def forward(self, x, mod=None):
        h = self.n1(x)
        if self.n_modalities and mod is not None:
            h = self._mod(self.f1, h, mod)
        h = self.c1(F.silu(h))
        h = self.n2(h)
        if self.n_modalities and mod is not None:
            h = self._mod(self.f2, h, mod)
        h = self.c2(F.silu(h))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, width=64, mults=(1, 2, 4), blocks_per_level=2,
                 t_in=T_IN, t_out=T_OUT, channels=C, n_modalities=0, cond="norm",
                 anchor="last"):
        super().__init__()
        # What the predicted delta is added to. `last` is the shipped behaviour and
        # the default, so every checkpoint packed before 2026-08-18 -- which is all
        # of them, including 887158 -- loads and runs exactly as before. `blend`
        # learns a per-output-frame mix of the last observed frame and the input
        # window's time-mean: t+1 is nearly persistence, t+20 nearly climatology,
        # and 20 scalars let the network say where the crossover is.
        self.anchor = anchor
        self.t_in, self.t_out, self.c = t_in, t_out, channels
        self.n_modalities = n_modalities
        self.cond = cond if n_modalities else "none"
        nb = n_modalities if self.cond == "norm" else 0
        cin, cout = t_in * channels, t_out * channels
        chans = [width * m for m in mults]

        self.stem = nn.Conv2d(cin, chans[0], 3, padding=1)
        self.downs, self.pools = nn.ModuleList(), nn.ModuleList()
        prev = chans[0]
        for i, ch in enumerate(chans):
            self.downs.append(nn.ModuleList(
                [Block(prev if b == 0 else ch, ch, n_modalities=nb)
                 for b in range(blocks_per_level)]))
            prev = ch
            self.pools.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                              if i < len(chans) - 1 else nn.Identity())
        self.mid1 = Block(prev, prev, n_modalities=nb)
        self.mid2 = Block(prev, prev, n_modalities=nb)
        self.ups, self.upsamples = nn.ModuleList(), nn.ModuleList()
        for i, ch in reversed(list(enumerate(chans))):
            self.upsamples.append(nn.ConvTranspose2d(prev, ch, 4, stride=2, padding=1)
                                  if i < len(chans) - 1 else nn.Identity())
            prev = ch if i < len(chans) - 1 else prev
            self.ups.append(nn.ModuleList(
                [Block(prev + (ch if b == 0 else 0), ch, n_modalities=nb)
                 for b in range(blocks_per_level)]))
            prev = ch
        self.out_norm = nn.GroupNorm(8, prev)
        self.out = nn.Conv2d(prev, cout, 3, padding=1)
        if anchor == "blend":
            self.anchor_logit = nn.Parameter(torch.zeros(t_out))
        if self.cond == "enc":
            self.mod_emb = nn.Embedding(n_modalities, width * 4)
            self.film = nn.ModuleList([nn.Linear(width * 4, 2 * ch) for ch in chans])

    def forward(self, x):
        # Always the real row: the simulated one exists only for training.
        idx = (torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
               if self.n_modalities else None)
        m = idx if self.cond == "norm" else None
        e = self.mod_emb(idx) if self.cond == "enc" else None
        h = self.stem(x)
        skips = []
        for i, (level, pool) in enumerate(zip(self.downs, self.pools)):
            for blk in level:
                h = blk(h, m)
            if e is not None:
                g, b = self.film[i](e).chunk(2, dim=-1)
                h = h * (1 + g[:, :, None, None]) + b[:, :, None, None]
            skips.append(h)
            h = pool(h)
        h = self.mid2(self.mid1(h, m), m)
        for level, up, skip in zip(self.ups, self.upsamples, reversed(skips)):
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            for blk in level:
                h = blk(h, m)
        y = self.out(F.silu(self.out_norm(h)))
        last = x[:, (self.t_in - 1) * self.c: self.t_in * self.c]
        if self.anchor != "blend":
            return y + last.repeat(1, self.t_out, 1, 1)
        b, _, h, w = x.shape
        wmean = x[:, :self.t_in * self.c].reshape(
            b, self.t_in, self.c, h, w).mean(dim=1)
        a = torch.sigmoid(self.anchor_logit).to(x.dtype)[None, :, None, None, None]
        base = (a * last[:, None] + (1 - a) * wmean[:, None])
        return y + base.reshape(b, self.t_out * self.c, h, w)


SIGMA_GLOBAL = 0.0563870259


class UncertaintyHead(nn.Module):
    def __init__(self, t=T_IN, c=C, width=48, mults=(1, 2, 4), extra_channels=0):
        super().__init__()
        self.t, self.c = t, c
        self.extra_channels = extra_channels
        cin = t * c * 2 + 1 + extra_channels
        chans = [width * m for m in mults]
        self.stem = nn.Conv2d(cin, chans[0], 3, padding=1)
        self.downs, self.pools = nn.ModuleList(), nn.ModuleList()
        prev = chans[0]
        for i, ch in enumerate(chans):
            self.downs.append(Block(prev, ch))
            prev = ch
            self.pools.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                              if i < len(chans) - 1 else nn.Identity())
        self.mid = Block(prev, prev)
        self.ups, self.upsamples = nn.ModuleList(), nn.ModuleList()
        for i, ch in reversed(list(enumerate(chans))):
            self.upsamples.append(nn.ConvTranspose2d(prev, ch, 4, stride=2, padding=1)
                                  if i < len(chans) - 1 else nn.Identity())
            prev = ch if i < len(chans) - 1 else prev
            self.ups.append(Block(prev + ch, ch))
            prev = ch
        self.out_norm = nn.GroupNorm(8, prev)
        self.out = nn.Conv2d(prev, 2 * t * c, 3, padding=1)

    def forward(self, window, pred, log_scale, extra=None):
        b, _, h, w = window.shape
        plane = log_scale.reshape(b, 1, 1, 1).expand(b, 1, h, w)
        parts = [window, pred, plane] + ([extra] if self.extra_channels else [])
        z = self.stem(torch.cat(parts, dim=1))
        skips = []
        for blk, pool in zip(self.downs, self.pools):
            z = blk(z)
            skips.append(z)
            z = pool(z)
        z = self.mid(z)
        for blk, up, skip in zip(self.ups, self.upsamples, reversed(skips)):
            z = up(z)
            z = blk(torch.cat([z, skip], dim=1))
        d = torch.exp(self.out(F.silu(self.out_norm(z))).clamp(-9.0, -2.0))
        n = self.t * self.c
        return d[:, :n], d[:, n:]          # widths below and above the prediction


class Combiner(nn.Module):
    """Learned replacement for the plain mean over the ensemble members.

    The members are trained independently and then averaged, but averaging is not
    neutral on the quantity TKE measures: each member approximates the conditional
    mean of a stochastic target, so their average carries the fluctuation of an
    average. Measured on this ensemble, the members hold 7.6% more TKE than their
    mean at the median cell and 164% more at p95, and tke is the weakest accuracy
    subscore by a factor of five in error.

    Rescaling the fluctuation back afterwards was tried and is dose-dependently
    negative -- it imposes a fixed trade against rel_l2 with no way to learn where
    it is worth making. This learns the trade instead, under the same composite
    loss the forecasters use, so the scorer's own weighting decides. The output
    convolution is zero-initialised at training time, so an untrained combiner is
    exactly the ensemble mean and every departure had to pay for itself.

    Priced before shipping: 0.097 ms/sample locally, ~0.17 in the container by the
    conversion factor submission 893273 measured, i.e. -0.009 of final against a
    measured +0.035 on held-out Reynolds numbers.
    """

    def __init__(self, n_members=3, width=64, depth=3, spread=True, window=True,
                 groups=8, t_out=T_OUT, t_in=T_IN, c=C):
        super().__init__()
        self.spread, self.window = spread, window
        cin = n_members * t_out * c + t_out * c
        if spread:
            cin += t_out * c
        if window:
            cin += t_in * c
        layers = [nn.Conv2d(cin, width, 3, padding=1)]
        for _ in range(max(0, depth - 1)):
            layers += [nn.GroupNorm(min(groups, width), width), nn.SiLU(),
                       nn.Conv2d(width, width, 3, padding=1)]
        layers += [nn.GroupNorm(min(groups, width), width), nn.SiLU()]
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv2d(width, t_out * c, 3, padding=1)

    def forward(self, members, window=None):
        mean = members.mean(0)
        parts = [members.permute(1, 0, 2, 3, 4).reshape(
            members.shape[1], -1, *members.shape[-2:]), mean]
        if self.spread:
            parts.append(members.std(0, unbiased=True))
        if self.window:
            parts.append(window)
        return mean + self.out(self.body(torch.cat(parts, dim=1)))


_state = {}


def _load():
    if not _state:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(_CKPT, map_location=device)
        # Weights are stored fp16 to fit three width-96 models under the 256 MB
        # cap; they are cast back to fp32 here. Verified lossless in the scored
        # metrics: subscores identical to four decimals.
        # Weights are stored fp16 and, on a GPU, stay fp16: the fields are O(1)
        # after per-window normalization and the outputs are cast to fp16 on the way
        # out regardless, so the only thing at stake is arithmetic -- and the scored
        # subscores come out unchanged. Casting up to fp32 was costing time for
        # nothing, and time_score's weight is ~0.09, not the ~0.00 an earlier fit
        # suggested. Off CUDA the weights are widened, where fp16 would be slower.
        dt = torch.float16 if device == "cuda" else torch.float32

        def f32(sd):
            return {k: (v.to(dt) if v.is_floating_point() else v) for k, v in sd.items()}
        # Half-precision modules in channels_last on the GPU. Submission 914270 showed
        # the container calls predict() one window at a time, so the clock is the GPU-side
        # chain of ~300 small kernels per member; 60-66 of those are the layout
        # conversions cuDNN inserts to reach its NHWC kernels, and channels_last removes
        # them. fp16 arithmetic on top trims the rest: end to end 4.12 -> 3.49 ms per call
        # on an A100 (docs: FINDINGS 9c). Outputs move by one to two fp16 ulps (2.4e-4);
        # the kit scorer's subscores are unchanged to three decimals. The cold-shape
        # penalty that made fp16 lose in 891095 does not apply: every shape is captured
        # at import. Off CUDA everything stays fp32 and contiguous.
        fmt = torch.channels_last if device == "cuda" else torch.contiguous_format
        models = []
        for sd in ck["models"]:
            # mults/blocks default to the shipped architecture when absent, so a
            # checkpoint packed before 2026-08-17 still loads unchanged.
            m = UNet(width=ck["width"], n_modalities=2 if ck["film"] else 0,
                     cond=ck.get("cond", "enc"),
                     mults=tuple(int(v) for v in str(ck.get("mults") or "1,2,4").split(",")),
                     blocks_per_level=int(ck.get("blocks") or 2),
                     anchor=ck.get("anchor") or "last")
            m.load_state_dict(f32(sd))
            models.append(m.to(device=device, dtype=dt, memory_format=fmt).eval())
        head = UncertaintyHead(width=ck["head_width"],
                               extra_channels=80 if ck.get("head_spread") else 0)
        head.load_state_dict(f32(ck["head"]))
        head = head.to(dtype=dt, memory_format=fmt)
        # Absent from every checkpoint built before 2026-08-19, so an older bundle
        # loads unchanged and predicts through the plain mean.
        comb = None
        if ck.get("combiner") is not None:
            comb = Combiner(**ck["combiner_cfg"])
            comb.load_state_dict(f32(ck["combiner"]))
            comb = comb.to(device).eval()
        _state.update(models=models, head=head.to(device).eval(), device=device,
                      combiner=comb)
    return _state["models"], _state["head"], _state["device"], _state["combiner"]


#: Shapes forwarded once at setup so cuDNN picks its algorithms and allocates its
#: workspaces there rather than during a timed call. Measured locally at 187 windows
#: in a fresh process: no warm-up 2.94 ms/sample, warming (1, 8, 64) 1.32, warming
#: the exact 187 1.17. Most of the cost is handles and algorithm choice rather than
#: per-N workspace, which is what keeps this honest -- the batch size never has to be
#: guessed and chunks are never padded. Padding was the half of 891095 that added
#: real work, and the rule that has survived every submission touching runtime is
#: that removing real work pays and moving it does not.
_WARM = (1, 8, 64)
_SETUP_DONE = []


def _setup():
    """Load and warm, once, outside any timed region. Idempotent.

    Doing this at *import* rather than in ``SubmissionModel.__init__`` is deliberate.
    ``competition_info/submission.md`` says the ingestion program "also supports" the
    class, without saying which entry point wins when a file offers both -- and this
    repository's own ``tools/run_and_score.py`` resolves that ambiguity by preferring
    the module-level ``predict``. If the real ingestion does the same, work parked in
    ``__init__`` would never run. Import happens before either entry point is called
    under both interfaces, so this is the one place that cannot be bypassed.

    Wrapped, because a submission that raises at import scores zero on every
    subscore. Any failure here falls back to the shipped lazy path, which is exactly
    the behaviour of every submission before this one.
    """
    if _SETUP_DONE:
        return
    _SETUP_DONE.append(True)
    try:
        _load()
        for k in _WARM:
            _predict_eager(np.zeros((k, T_IN, 32, 64, 3), dtype=np.float32))
        _build_graphs()
    except Exception:      # noqa: BLE001 -- never let setup break inference
        pass


def _window_scale(uv):
    """RMS speed over the input window's valid cells, per window, on device.

    Cells inside the airfoil body and outside the PIV field of view are stored as
    exact zeros. They contribute nothing to a sum of squares, so only the count of
    valid cells needs the mask -- and it must be the window-local mask, since a
    single window is all we ever see.
    """
    tot = uv.pow(2).sum(dim=(1, 2, 3, 4))
    nz = (uv.abs() >= EPS).any(dim=-1).any(dim=1)
    return (tot / (T_IN * nz.sum(dim=(1, 2)).clamp_min(1))).sqrt() + EPS


def predict(input_array, metadata=None):
    """(N, T_in, H, W, C) -> dict with prediction (N, T_out, H, W, C) plus bounds."""
    x = np.asarray(input_array, dtype=np.float32)
    if _GRAPHS and x.ndim == 5 and x.shape[1:] == (T_IN, 32, 64, 3):
        try:
            return _predict_graphed(x)
        except Exception:  # noqa: BLE001 -- fall through to the eager path
            pass
    return _predict_eager(x)


def _predict_eager(x):
    """The eager path: submission 912717's predict(), unchanged."""
    n, _, h, w, _ = x.shape
    models, head, device, combiner = _load()

    # Accumulate in fp16, which is what leaves the container anyway. Building these
    # in fp32 and calling .half() at the end costs a full-size allocation plus a
    # conversion pass over three arrays -- profiled at 34% of predict(), against 5%
    # for the device-to-host copy itself. The copy was never the expensive part.
    out_dt = torch.float16 if device == "cuda" else torch.float32
    pred = torch.empty((n, T_OUT, h, w, 3), dtype=out_dt, device=device)
    lo = torch.empty_like(pred)
    hi = torch.empty_like(pred)

    # Half precision on the GPU. The weights already live in fp16 in the archive,
    # the fields are O(1) after per-window normalization, and the outputs are cast
    # to fp16 on the way out regardless -- so the only thing at stake is arithmetic,
    # and the scored subscores are unchanged to four decimals. It matters because
    # time_score's weight is ~0.09, not the ~0.00 an earlier fit suggested: shipping
    # 1.5 ms/sample of avoidable work cost 0.66 of that subscore on 2026-08-10.
    # Skipped entirely off CUDA, where fp16 would be slower rather than faster.
    with torch.no_grad():
        for i in range(0, n, _BATCH):
            xb = torch.from_numpy(x[i:i + _BATCH]).to(device, non_blocking=True)
            uv = xb[..., :2]
            scale = _window_scale(uv)
            s = scale[:, None, None, None, None]
            p0 = next(models[0].parameters())
            fmt = (torch.channels_last if p0.is_contiguous(memory_format=torch.channels_last)
                   and p0.dim() == 4 else torch.contiguous_format)
            xin = (uv / s).permute(0, 1, 4, 2, 3).reshape(-1, T_IN * C, h, w)
            xin = xin.to(p0.dtype).contiguous(memory_format=fmt)

            # Back to fp32 for the statistics: the spread is a difference of nearby
            # numbers and the interval widths are compared against sigma_global in
            # absolute terms, neither of which wants half precision.
            outs = torch.stack([m(xin) for m in models]).float()
            # The combiner reads the members and the normalised input window, and
            # returns a correction to their mean. The spread fed to the interval
            # head below stays the *members'* disagreement either way: the combiner
            # changes the prediction, not the evidence about how uncertain it is.
            y = outs.mean(0) if combiner is None else combiner(outs, xin.float())
            extra = None
            if head.extra_channels:
                # De-normalized to physical units and divided by sigma_global, so
                # the spread reaches the head on the same footing as the widths it
                # emits. Both the ratio and its log go in: it spans two orders of
                # magnitude between quiescent cells and the wake.
                r = outs.std(0, unbiased=True) * scale[:, None, None, None] / SIGMA_GLOBAL
                extra = torch.cat([r, torch.log(r.clamp_min(1e-4))], dim=1)
            hdt = next(head.parameters()).dtype
            below, above = head(xin, y.to(hdt), torch.log(scale).to(hdt),
                                None if extra is None else extra.to(hdt))
            below, above = below.float(), above.float()
            y_plain = y
            if GAIN_DOSE > 0 and combiner is None:
                y = _matched_gain(outs, y)

            yy = y.reshape(-1, T_OUT, C, h, w).permute(0, 1, 3, 4, 2) * s
            yp = y_plain.reshape(-1, T_OUT, C, h, w).permute(0, 1, 3, 4, 2) * s
            bb = below.reshape(-1, T_OUT, C, h, w).permute(0, 1, 3, 4, 2)
            aa = above.reshape(-1, T_OUT, C, h, w).permute(0, 1, 3, 4, 2)
            sl = slice(i, i + _BATCH)
            pred[sl, ..., :2] = yy.to(out_dt)
            pred[sl, ..., 2] = 0.0                       # p is unmeasured
            # Interval widths are absolute, as sigma_global is, so they are not
            # rescaled by the window scale.
            lo[sl, ..., :2] = (yp - bb).to(out_dt)      # intervals bracket the PLAIN mean (k4's head)
            lo[sl, ..., 2] = 0.0
            hi[sl, ..., :2] = (yp + aa).to(out_dt)
            hi[sl, ..., 2] = 0.0

    # A non-finite prediction, or a non-finite or reversed bound, zeroes every
    # subscore, so guarantee it here rather than trusting the forward pass.
    for t in (pred, lo, hi):
        torch.nan_to_num_(t, nan=0.0, posinf=0.0, neginf=0.0)
    torch.minimum(lo, pred, out=lo)
    torch.maximum(hi, pred, out=hi)

    return {
        # k4a2f32 (2026-09-18): the fp16 values are kept (bit-identical decoded outputs to 929146) and upcast on the
        # GPU before the host copy, so the harness receives float32 arrays -- the payload-cost hypothesis of
        # CANDIDATE_timing_p0f32.md applied to the trained artefact. No other change.
        "prediction": pred.float().cpu().numpy(),
        "lower": lo.float().cpu().numpy(),
        "upper": hi.float().cpu().numpy(),
    }

#: Batch shapes captured as CUDA graphs at setup. Up to eight windows the three members
#: replay concurrently on their own streams (the GPU is far from saturated, so the
#: three forward passes overlap); from sixteen up one sequential graph is faster. A
#: call of k windows uses the smallest bucket >= k, padding by repeating its last
#: window; calls above the largest bucket are chunked through it. Measured on an
#: A100 (the evaluation card is A800, same silicon) and an H100: 2.5x at one window,
#: 2x at two to three, 1.4x at eight, neutral at 64 -- so the change cannot lose on
#: any call size and pays where the container actually is.
_GRAPH_PAR = (1, 2, 3, 4, 6, 8)
_GRAPH_SEQ = (16, 32, 64)
_GRAPHS = {}
_GRAPH_BUDGET_S = 40.0     # the prediction stage has a 180 s wall; capture is ~4 s


def _pre(xb, models):
    """Per-window normalisation and channel packing, exactly as the eager path."""
    n, _, h, w, _ = xb.shape
    uv = xb[..., :2]
    scale = _window_scale(uv)
    s = scale[:, None, None, None, None]
    xin = (uv / s).permute(0, 1, 4, 2, 3).reshape(-1, T_IN * C, h, w)
    p0 = next(models[0].parameters())
    fmt = torch.channels_last if p0.is_contiguous(memory_format=torch.channels_last) and p0.dim() == 4 else torch.contiguous_format
    return xin.to(p0.dtype).contiguous(memory_format=fmt), scale, s, h, w, n


def _post(xb, outs, models, head):
    """Mean, interval head and output assembly, exactly as the eager path."""
    xin, scale, s, h, w, n = _pre(xb, models)
    outs = outs.float()
    y = outs.mean(0)
    extra = None
    if head.extra_channels:
        r = outs.std(0, unbiased=True) * scale[:, None, None, None] / SIGMA_GLOBAL
        extra = torch.cat([r, torch.log(r.clamp_min(1e-4))], dim=1)
    hdt = next(head.parameters()).dtype
    below, above = head(xin, y.to(hdt), torch.log(scale).to(hdt),
                        None if extra is None else extra.to(hdt))
    below, above = below.float(), above.float()
    y_plain = y
    if GAIN_DOSE > 0:
        y = _matched_gain(outs, y)
    yy = y.reshape(-1, T_OUT, C, h, w).permute(0, 1, 3, 4, 2) * s
    yp = y_plain.reshape(-1, T_OUT, C, h, w).permute(0, 1, 3, 4, 2) * s
    bb = below.reshape(-1, T_OUT, C, h, w).permute(0, 1, 3, 4, 2)
    aa = above.reshape(-1, T_OUT, C, h, w).permute(0, 1, 3, 4, 2)
    pred = torch.zeros((n, T_OUT, h, w, 3), dtype=torch.float16, device=xb.device)
    lo = torch.zeros_like(pred)
    hi = torch.zeros_like(pred)
    pred[..., :2] = yy.to(pred.dtype)
    lo[..., :2] = (yp - bb).to(pred.dtype)
    hi[..., :2] = (yp + aa).to(pred.dtype)
    for t in (pred, lo, hi):
        torch.nan_to_num_(t, nan=0.0, posinf=0.0, neginf=0.0)
    return pred, torch.minimum(lo, pred), torch.maximum(hi, pred)


def _capture(fn, *args):
    """Warm once on a side stream, then capture. Returns (graph, static outputs)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(*args)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn(*args)
    return g, out


class _Graphed:
    """One batch shape, captured. Each member graph owns its memory pool, so the
    three can replay concurrently without sharing intermediate buffers; the post
    graph reads their static outputs after the streams are joined."""

    def __init__(self, n, models, head, device, parallel):
        self.n, self.parallel = n, parallel
        self.static = torch.zeros((n, T_IN, 32, 64, 3), device=device)
        if parallel:
            self.streams = [torch.cuda.Stream() for _ in models]
            self.gm, mout = [], []
            for m in models:
                g, o = _capture(lambda xb, m=m: m(_pre(xb, models)[0]), self.static)
                self.gm.append(g)
                mout.append(o)
            self.gp, self.out = _capture(
                lambda xb: _post(xb, torch.stack(mout), models, head), self.static)
        else:
            self.g, self.out = _capture(
                lambda xb: _post(xb, torch.stack([m(_pre(xb, models)[0]) for m in models]),
                                 models, head), self.static)

    def run(self, x_np):
        k = x_np.shape[0]
        self.static[:k].copy_(torch.from_numpy(np.ascontiguousarray(x_np)))
        if k < self.n:      # pad by repeating the last window; the rows are sliced off
            self.static[k:].copy_(self.static[k - 1:k].expand(self.n - k, -1, -1, -1, -1))
        if self.parallel:
            cur = torch.cuda.current_stream()
            for g, s in zip(self.gm, self.streams):
                s.wait_stream(cur)
                with torch.cuda.stream(s):
                    g.replay()
            for s in self.streams:
                cur.wait_stream(s)
            self.gp.replay()
        else:
            self.g.replay()
        return [o[:k] for o in self.out]


def _build_graphs():
    """Capture every bucket, then check two of them against the eager path. Any
    failure leaves `_GRAPHS` empty and predict() on the eager path, which is exactly
    submission 912717."""
    import time
    models, head, device, combiner = _load()
    if device != "cuda" or combiner is not None:
        return
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            for b in sorted(set(_GRAPH_PAR) | set(_GRAPH_SEQ)):
                if time.perf_counter() - t0 > _GRAPH_BUDGET_S:
                    raise RuntimeError("capture budget exceeded")
                _GRAPHS[b] = _Graphed(b, models, head, device, parallel=b in _GRAPH_PAR)
            # Self-check: the smallest and the largest bucket against eager, on data
            # with the statistics of a real window (random fields would not exercise
            # the normalisation the same way, but the check is about the plumbing).
            rng = np.random.default_rng(0)
            for b in (min(_GRAPHS), max(_GRAPHS)):
                x = rng.normal(0.15, 0.05, size=(b, T_IN, 32, 64, 3)).astype(np.float32)
                x[..., 2] = 0.0
                ref = _predict_eager(x)
                got = _GRAPHS[b].run(x)
                for key, o in zip(("prediction", "lower", "upper"), got):
                    d = float(np.abs(o.cpu().numpy().astype(np.float32)
                                     - ref[key].astype(np.float32)).max())
                    if not d < 2e-3:
                        raise RuntimeError(f"graph self-check failed at bucket {b}: {d}")
            torch.cuda.synchronize()
    except Exception:      # noqa: BLE001 -- the eager path is the known-good fallback
        _GRAPHS.clear()


def _pieces(n):
    """Split a call of n windows into (bucket, count) replays.

    Up to eight windows: one bucket, padded by at most two windows -- the GPU is
    latency-bound there and the padding is free. Above eight: an exact-sum greedy
    decomposition (64s, 32s, 16s, then one small bucket), because padding a
    compute-bound batch is real work -- measured, padding 33 -> 64 ran 0.6x eager,
    while 32 + 1 is neutral. So the change cannot lose on any call size.
    """
    small = sorted(b for b in _GRAPHS if b <= 8)
    large = sorted((b for b in _GRAPHS if b > 8), reverse=True)
    out, k = [], n
    while k > 0:
        if k > 8 and large:
            b = next((b for b in large if b <= k), None)
            if b is None:            # 9..15: the biggest small bucket, then the rest
                b = small[-1]
            out.append((b, min(b, k)))
        else:
            b = next((b for b in small if b >= k), None)
            if b is None:            # only if no small bucket covers k (never, in practice)
                b = large[-1]
            out.append((b, min(b, k)))
        k -= out[-1][1]
    return out


def _predict_graphed(x):
    parts = {"prediction": [], "lower": [], "upper": []}
    i = 0
    with torch.no_grad():
        for b, k in _pieces(x.shape[0]):
            out = _GRAPHS[b].run(x[i:i + k])
            for key, o in zip(parts, out):
                parts[key].append(o.float().cpu().numpy())      # k4a2f32: float32 payload, fp16 values
            i += k
    return {k: (v[0] if len(v) == 1 else np.concatenate(v)) for k, v in parts.items()}


_setup()


class SubmissionModel:
    """Class interface, so construction and loading leave the timed region.

    The FAQ is explicit: "Model construction and checkpoint loading run outside the
    timed region; only inference is timed" -- and until now this file loaded lazily
    inside ``predict``, so the CUDA context, the weight transfers and cuDNN's
    per-shape workspaces were all being charged to ``time_score``. ``submission.md``
    documents ``SubmissionModel.__init__(submission_dir, device)`` as an accepted
    entry point; moving ``_load()`` there is the whole change.

    Both entry points stay live on purpose. If the ingestion program uses this
    class, the setup cost moves out; if it only looks for the module-level
    ``predict``, behaviour is byte-for-byte what it was. That way the wrapper cannot
    fail closed, which matters because the Decision Phase re-runs this code.

    Nothing else is touched. In particular the dtype stays as it was and chunks are
    not padded to bucket shapes: submission 891095 bundled those two with an
    init-time load and read -0.39 ms/sample on the board, which FINDINGS section 9
    took as evidence that the container never charged setup to the timed region.
    That inference does not survive the 2026-09-01 measurement that an honest fp16
    cast costs +15-16% *cold* (19.64 -> 22.68 ms/sample in a fresh process), which
    covers the whole regression on its own. So the init-time move has never actually
    been measured alone, and this is that measurement.
    """

    def __init__(self, submission_dir=None, device="cpu"):
        # The checkpoint path is resolved relative to this file, so the argument is
        # accepted and ignored rather than trusted; _CKPT is what the archive ships.
        self._dir = submission_dir
        _setup()          # idempotent, and normally already done at import

    def load_checkpoint(self, path=None, device=None):
        """Accepted because submission.md says the ingestion calls it when
        ``model.pth`` exists. ``_load()`` is idempotent, so this is a no-op after
        ``__init__`` and a load if the ingestion calls it first."""
        _load()

    def predict(self, input_array, metadata=None):
        return predict(input_array, metadata)
