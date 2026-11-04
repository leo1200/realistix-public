"""Wavenumber split of the 32x64 evaluation grid.

Measured on held-out real data, a U-Net's predictions decompose sharply by scale:
coherence with the target is 0.76-0.78 for |k| in 2-4, falls through 0.54 at 6-8,
and is 0.25-0.38 above 8. Above the crossover the model retains almost no phase
information and destroys 75-91% of the energy -- it is not failing so much as
correctly declining to predict something unpredictable, then paying for the
missing variance in the TKE subscore.

So the field is split once, exactly and invertibly, at ``K_CUT``:

* ``|k| <  K_CUT`` is forecast deterministically -- phase is recoverable there;
* ``|k| >= K_CUT`` is *generated* -- only its statistics are recoverable.

An FFT split has no reconstruction floor, which is why it is preferred here over
an autoencoder bottleneck: at useful compression an AE already costs a TKE error
of 0.22-0.32 against a target of ~0.50, spending most of the budget before the
generative part contributes anything.
"""

from __future__ import annotations

import torch

H, W = 32, 64
K_CUT = 6.0


def radial_k(h: int = H, w: int = W, device=None) -> torch.Tensor:
    """|k| on the rfft2 grid, in cycles per domain."""
    ky = torch.fft.fftfreq(h, device=device) * h
    kx = torch.fft.rfftfreq(w, device=device) * w
    return torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)


_cache: dict = {}


def masks(k_cut: float = K_CUT, h: int = H, w: int = W, device=None):
    key = (k_cut, h, w, str(device))
    if key not in _cache:
        kr = radial_k(h, w, device)
        low = (kr < k_cut).to(torch.float32)
        _cache[key] = (low, 1.0 - low)
    return _cache[key]


def split(x, k_cut: float = K_CUT):
    """(B, C, H, W) -> (low, high), exactly additive: low + high == x."""
    h, w = x.shape[-2:]
    low_m, high_m = masks(k_cut, h, w, x.device)
    f = torch.fft.rfft2(x, norm="ortho")
    low = torch.fft.irfft2(f * low_m, s=(h, w), norm="ortho")
    high = torch.fft.irfft2(f * high_m, s=(h, w), norm="ortho")
    return low.to(x.dtype), high.to(x.dtype)


def lowpass(x, k_cut: float = K_CUT):
    return split(x, k_cut)[0]


def highpass(x, k_cut: float = K_CUT):
    return split(x, k_cut)[1]


def project_high(x, k_cut: float = K_CUT):
    """Force a field to live entirely above the cut.

    The flow's samples must not leak into the deterministic branch's band --
    that leakage is precisely the failure mode the split exists to prevent.
    """
    return highpass(x, k_cut)


# --------------------------------------------------------------------------
# Lead-time-dependent split.
#
# A single k_cut is the wrong shape for this problem. Coherence decays in both
# scale and lead time, and much faster in scale at long lead: at t+1 every band
# down to |k|=12-33 is still predicted at 0.69, while by t+4-6 that same band is
# at 0.20. The 0.5-coherence contour therefore sweeps from k>33 at t+1 down to
# k~5 at t+16-20. Splitting on measured coherence instead of a fixed k beats the
# fixed cut on rel-L2 *and* TKE while generating fewer modes.
#
# The mask is measured on the training split only.
# --------------------------------------------------------------------------

def load_coherence(path: str, device=None) -> torch.Tensor:
    import numpy as np
    return torch.from_numpy(np.load(path)).to(device=device, dtype=torch.float32)


def mask_from_coherence(coh: torch.Tensor, threshold: float = 0.5,
                        k_floor: float = 3.0) -> torch.Tensor:
    """(T, H, W_r) float mask: 1 where the flow generates, 0 where the U-Net predicts.

    Not a per-cell threshold. Predictability is physically monotone in |k| -- small
    scales decorrelate first -- but a per-cell estimate is noisy, and thresholding
    it directly is unstable exactly where it matters. The lowest wavenumbers happen
    to have mediocre coherence (0.53-0.73, they carry slow drift), so a raw
    threshold flips them into the generated set at some lead times and hands the
    flow a large share of the field's energy: measured generated std jumped to 0.44
    at t+10 against 0.055 at t+5.

    So coherence is radially binned, and each lead time gets a single cutoff --
    the smallest |k| beyond which the band-mean coherence stays under threshold.
    ``k_floor`` additionally protects the energy-carrying large scales, which must
    be forecast rather than invented whatever the estimate says.
    """
    t, h, wr = coh.shape
    kr = radial_k(h, (wr - 1) * 2, coh.device)
    kmax = int(kr.max().item()) + 1
    mask = torch.zeros_like(coh)
    for i in range(t):
        cut = kmax
        for k in range(kmax, int(k_floor) - 1, -1):
            band = (kr >= k - 1) & (kr < k)
            if band.any() and coh[i][band].mean() < threshold:
                cut = k - 1
            else:
                break
        mask[i] = (kr >= max(cut, k_floor)).to(coh.dtype)
    return mask



def split_masked(x, gen_mask, t: int = 20, c: int = 2):
    """Split (B, T*C, H, W) into (deterministic, generated) parts by a (T,H,W_r) mask.

    Exactly additive, like the scalar-cut version.
    """
    b, _, h, w = x.shape
    f = torch.fft.rfft2(x.reshape(b, t, c, h, w), norm="ortho")
    m = gen_mask.to(f.device)[None, :, None]
    det = torch.fft.irfft2(f * (1 - m), s=(h, w), norm="ortho").reshape(b, t * c, h, w)
    gen = torch.fft.irfft2(f * m, s=(h, w), norm="ortho").reshape(b, t * c, h, w)
    return det.to(x.dtype), gen.to(x.dtype)


def project_masked(x, gen_mask, t: int = 20, c: int = 2):
    """Keep only the generated modes -- the flow's output must not touch the rest."""
    return split_masked(x, gen_mask, t, c)[1]
