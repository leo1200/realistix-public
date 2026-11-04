"""Score a model with the leaderboard's own metric code during training.

Validation loss is not the objective here -- ``final_score`` is, and it is built
from subscores that behave nothing like MSE. TKE in particular is bounded below
at 66.67 for any mean-reverting prediction, so a run whose MSE is still falling
can be flat on the metric that separates the field. Watching the real subscores
from the start is cheaper than discovering that at submission time.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

_KIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "starting_kit")
if _KIT not in sys.path:
    sys.path.insert(0, _KIT)
import scoring as S  # noqa: E402

T_OUT = 20


def to_ntchw(flat, t=T_OUT, c=2):
    """(B, T*C, H, W) -> (B, T, H, W, 3) with p zeroed, the scorer's layout."""
    b, _, h, w = flat.shape
    x = flat.reshape(b, t, c, h, w).permute(0, 1, 3, 4, 2)
    return torch.cat([x, torch.zeros_like(x[..., :1])], dim=-1)


@torch.no_grad()
def amp_dtype_for(device) -> torch.dtype:
    """bf16 where the card supports it, fp16 otherwise.

    Turing and earlier have no bfloat16, and torch raises rather than falling back --
    which turned a figure job on an idle RTX 2080 Ti into a crash while every A100 in
    the cluster was busy. Inference here is numerically undemanding either way; the
    shipped submission does not go through this path at all -- it loads fp16-rounded
    weights into fp32 modules and runs them as **fp32** on GPU, because
    `load_state_dict` preserves the destination dtype. Verified 2026-09-01: 488/488
    live parameters are float32 after a real `predict()`. (`docs/archive/FINDINGS_2026-09-10.md` section 9,
    where this was twice recorded the other way round.)
    """
    if str(device) == "cpu" or not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@torch.no_grad()
def predict_all(model, loader, device, amp_dtype=None):
    preds, tgts = [], []
    amp_dtype = amp_dtype or amp_dtype_for(device)
    model.eval()
    # Regime labels for models that condition on them. The loader does not shuffle,
    # so slicing the dataset's own ordering batch by batch is exact.
    regimes = getattr(loader.dataset, "window_regimes", None)
    at = 0
    for x, y, scale in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
            # log_scale is the Reynolds cue that per-window normalisation removes;
            # models that do not condition on it ignore the argument.
            r = None
            if regimes is not None:
                r = torch.from_numpy(regimes[at:at + x.shape[0]]).to(device)
            at += x.shape[0]
            out = model(x, None, torch.log(scale.to(device)), r)
        if isinstance(out, tuple):
            out = out[0]        # drop the auxiliary pressure head
        s = scale.to(device)[:, None, None, None]
        preds.append((to_ntchw(out.float() * s)).cpu().numpy())
        tgts.append((to_ntchw(y.to(device).float() * s)).cpu().numpy())
    return np.concatenate(preds).astype(np.float32), np.concatenate(tgts).astype(np.float32)


def subscores(pred, target, ms_per_sample=None, band_frac=0.15):
    c = S.measured_channels(target)
    rel = float(np.mean(S.rel_l2_per_sample(pred, target, c)))
    tke = float(np.mean(S.tke_rel_l2_per_sample(pred, target, c)))
    mvpe = S.mvpe_rel_l2(pred, target)

    hw = band_frac * S.SIGMA_GLOBAL
    sps, cov = S.aggregate_sps(pred, target, c,
                               pred[..., :c] - hw, pred[..., :c] + hw)

    out = {
        "rel_l2_score": S.score_error(rel),
        "tke_score": S.score_error(tke),
        "mvpe_score": S.score_error(mvpe),
        "sps_score": S.score_sps(sps),
        "_rel_l2": rel, "_tke": tke, "_mvpe": mvpe, "_coverage": cov,
    }
    if ms_per_sample is not None:
        out["time_score"] = S.score_time(ms_per_sample / 1000.0)
    return out


def fmt(sc):
    keys = [k for k in ("rel_l2_score", "tke_score", "mvpe_score", "time_score", "sps_score")
            if k in sc]
    mean = np.mean([sc[k] for k in keys])
    body = "  ".join(f"{k.replace('_score', '')}={sc[k]:.2f}" for k in keys)
    return f"{body}  | mean={mean:.2f}  (relL2={sc['_rel_l2']:.4f} tke={sc['_tke']:.3f})"
