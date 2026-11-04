"""A Fourier neural operator over the same 32x64 windows, as an ensemble member.

Why a second architecture at all. Ensembling three U-Nets that differ only in
seed, loss weights or schedule length measured flat: +0.02 to +0.07 on `val_aoa`
against the seed-only ensemble, i.e. nothing. They are the same function class
trained on the same data, so they make the same mistakes, and averaging correlated
mistakes does not remove them. That matters twice over now, because the interval
head reads the members' disagreement directly -- an ensemble that agrees for
structural reasons reports confidence it has not earned.

An FNO disagrees structurally. It mixes globally at every layer through a
truncated Fourier basis instead of stacking local receptive fields, it never
downsamples, and its inductive bias is toward smooth large-scale structure rather
than edges. RealPDEBench ranks it below a U-Net on this scenario and the
organizers' own FNO reaches rel-L2 96.6 in-sample, so it is a weaker but not weak
model -- which is the right profile for a minority ensemble member.

Everything that was load-bearing for the U-Net is kept, so the comparison isolates
the architecture:

* predicting a delta on the last input frame from a zero-initialised projection,
  so training starts exactly at persistence;
* FiLM conditioning on modality, so it can train jointly on real and simulated
  windows and be told which it is looking at;
* time folded into channels, since the horizons are fixed at 20.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

T_IN = T_OUT = 20
C = 2  # u, v


class SpectralConv2d(nn.Module):
    """Truncated Fourier multiplication: rfft2 -> keep low modes -> irfft2.

    The kept band is ``modes1`` signed vertical wavenumbers by ``modes2`` real
    horizontal ones. Predictability is strongly scale-dependent here -- coherence
    is 0.90 at k in [2, 6] and 0.16 at k > 12 -- so truncation is not only a cost
    control, it discards the part of the spectrum the model could not forecast
    anyway.
    """

    def __init__(self, cin: int, cout: int, modes1: int, modes2: int):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (cin * cout)
        # Two blocks: the positive and negative vertical wavenumbers. rfft2 leaves
        # the first axis full and halves the second, so the negative half of axis 1
        # is a distinct band and needs its own weights.
        # Keep the historical complex initialization/RNG order, but store both
        # components as real parameters. Muon flattens these as (cin, -1), so
        # its real bf16 Newton-Schulz path updates real AND imaginary channels.
        self.w1 = nn.Parameter(torch.view_as_real(
            scale * torch.rand(cin, cout, modes1, modes2, dtype=torch.cfloat)))
        self.w2 = nn.Parameter(torch.view_as_real(
            scale * torch.rand(cin, cout, modes1, modes2, dtype=torch.cfloat)))

    def forward(self, x):
        b, _, h, w = x.shape
        # Autocast hands this bf16 tensors; cuFFT and complex matmul want fp32.
        xf = torch.fft.rfft2(x.float(), norm="ortho")
        out = torch.zeros(b, self.w1.shape[1], h, w // 2 + 1,
                          dtype=torch.cfloat, device=x.device)
        w1 = torch.view_as_complex(self.w1.float())
        w2 = torch.view_as_complex(self.w2.float())
        m1 = min(self.modes1, h // 2)
        m2 = min(self.modes2, w // 2 + 1)
        out[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, :m1, :m2], w1[:, :, :m1, :m2])
        out[:, :, -m1:, :m2] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, -m1:, :m2], w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out, s=(h, w), norm="ortho")


class FNOBlock(nn.Module):
    """Global spectral mixing plus a pointwise channel mix, residually."""

    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spec = SpectralConv2d(width, width, modes1, modes2)
        self.pw = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(8, width)

    def forward(self, x):
        h = self.norm(x)
        return x + F.gelu(self.spec(h) + self.pw(h))


class FNO(nn.Module):
    def __init__(self, width: int = 96, layers: int = 4, modes1: int = 16,
                 modes2: int = 17, t_in: int = T_IN, t_out: int = T_OUT,
                 channels: int = C, predict_delta: bool = True,
                 n_modalities: int = 0):
        super().__init__()
        self.t_in, self.t_out, self.c = t_in, t_out, channels
        self.predict_delta = predict_delta
        self.n_modalities = n_modalities
        # Two coordinate planes alongside the window. The domain is fixed and the
        # airfoil's zero-mask moves with angle of attack, so absolute position is
        # informative -- and a spectral operator is otherwise translation-equivariant
        # and cannot represent "near the leading edge" at all.
        self.lift = nn.Conv2d(t_in * channels + 2, width, 1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes1, modes2)
                                     for _ in range(layers)])
        self.proj1 = nn.Conv2d(width, 2 * width, 1)
        self.proj2 = nn.Conv2d(2 * width, t_out * channels, 1)
        nn.init.zeros_(self.proj2.weight)
        nn.init.zeros_(self.proj2.bias)
        if n_modalities:
            self.mod_emb = nn.Embedding(n_modalities, width)
            self.film = nn.ModuleList([nn.Linear(width, 2 * width)
                                       for _ in range(layers)])
            for lin in self.film:            # start as the identity transform
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)

    def _grid(self, x):
        b, _, h, w = x.shape
        ys = torch.linspace(0, 1, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(0, 1, w, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gy, gx]).expand(b, 2, h, w)

    def forward(self, x, modality=None, log_scale=None, regime=None):
        """(B, T_in*C, H, W) -> (B, T_out*C, H, W), both in normalized units.

        ``regime`` is accepted and ignored: this operator conditions only on the
        modality token. Every caller passes it positionally (train_mixed's loss
        path, evaluate.predict_all), so omitting it made `--arch fno` raise a
        TypeError on its first batch rather than train a worse model.
        """
        e = None
        if self.n_modalities:
            if modality is None:
                modality = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            e = self.mod_emb(modality)
        h = self.lift(torch.cat([x, self._grid(x)], dim=1))
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if e is not None:
                g, b = self.film[i](e).chunk(2, dim=-1)
                h = h * (1 + g[:, :, None, None]) + b[:, :, None, None]
        y = self.proj2(F.gelu(self.proj1(h)))
        if self.predict_delta:
            last = x[:, (self.t_in - 1) * self.c: self.t_in * self.c]
            y = y + last.repeat(1, self.t_out, 1, 1)
        return y
