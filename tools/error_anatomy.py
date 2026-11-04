#!/usr/bin/env python3
"""Where in (component, k, omega) space does a training change move the error?

Built for the --w-sps question but general. The dose ladder showed `sps_bulk`
improving rel_l2 and the bulk of |e| while paying TKE, monotone in dose -- the
frontier signature. Whether that trade is *intrinsic* or an artefact of the term's
shape depends on where the two effects live, and the scorer itself defines the
decomposition that answers it:

    kinetic_energy() is the temporal variance about each signal's own time-mean,
    per pixel. So the prediction's TIME-MEAN is invisible to TKE by construction.

Split every error into e = e_mean + e_fluct (per-pixel time-mean error, and the
fluctuation about it -- orthogonal over t, so MSE decomposes exactly). Then:

  * If the --w-sps gain sits in e_mean while the TKE damping sits in the
    fluctuation amplitude, the two are SEPARABLE: a bulk term applied to the
    time-mean alone has no direct gradient path to predicted TKE at all, and the
    trade was an unwanted side effect of applying one term to both components.
  * If the gain is mostly var(e_fluct) falling because predicted fluctuation
    amplitude fell -- damping uncorrelated fluctuation lowers |e| wherever the
    model's phase is wrong -- then the gain IS the damping, the components are the
    same thing, and the frontier holds a fifth time.

Reported per arm, paired against a reference per fold:

  MSE split          mse_mean / mse_fluct, and the delta in each
  mean error by k    2D spectrum of e_mean:   k<2, 2-8, >=8
  fluct error by     3D spectrum of e_fluct:  omega in [1,2] (shedding/meandering
    (omega, k)       band) against omega>=3, crossed with k<8 / k>=8
  amplitude ratio    energy of predicted fluctuation over the target's, by k --
                     the direct measure of where damping happens (TKE deficit)
  bulk counterfactuals  coverage P(|e| <= h) at the R_const-optimal band h for the
                     full error, for e_fluct alone (mean error removed) and for
                     e_mean alone (fluctuation error removed) -- which channel is
                     carrying the bulk improvement

    python tools/error_anatomy.py --arms mu02 spsw05 spsw20 spsw80 --ref mu02
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import functools
import statistics
import sys

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "starting_kit"))
sys.path.insert(0, HERE)

import scoring as S  # noqa: E402
from kfold import MIMIC_FOLDS  # noqa: E402
import foldenv  # noqa: E402  -- must precede the realistix imports below

# splits.py freezes REALISTIX_HELDOUT_* at import, so the fold has to be in the
# environment before the next line runs. Until 2026-09-01 this tool set them
# inside main() instead, which meant --fold did nothing and every fold scored the
# default AoA=20 holdout. splits.partition now raises if they drift after import.
foldenv.apply_fold_from_argv()

from realistix.data import build  # noqa: E402
from realistix.evaluate import amp_dtype_for, subscores, to_ntchw  # noqa: E402
from train_final_head import Ensemble  # noqa: E402

print = functools.partial(print, flush=True)  # noqa: A001

# Fixed 2026-09-02. This tool used to loop the four mimic folds in one process and
# set REALISTIX_HELDOUT_* inside gather(), after realistix.splits had frozen them, so
# --fold was inert and every fold scored the default AoA=20 holdout. The tell is in the
# archived caches: all 24 of them carry the identical target amplitude vector
# (386.19727, 2629.2297, 463.26038) to eight significant figures, because the target
# set never changed -- only the fold-trained model did. FINDINGS.md section 10.
#
# The fix is the pattern kfold.py has always used: the parent spawns one child per
# (arm, fold) with the holdout in the child's environment, and only reads caches
# itself. The cache name carries a v2 so the pre-fix files cannot be picked up
# silently; they are archived under runs/anatomy_v1_prefold/.


RUNS = os.environ.get("REALISTIX_RUNS", "runs")
T = 20
#: The R_const-optimal half-width from tools/spread_info.py on these folds.
H_BULK = 0.124 * S.SIGMA_GLOBAL


def cells(fold: str) -> str:
    return next(f["env"]["REALISTIX_HELDOUT_CELLS"]
                for f in MIMIC_FOLDS if f["name"] == fold)


@torch.no_grad()
def gather(run_glob: str, fold: str, device: str):
    """(pred, target) in physical units, (N, T, C, H, W), on the fold's holdout."""
    for k, v in (("REALISTIX_HELDOUT_AOA", ""), ("REALISTIX_HELDOUT_RE", ""),
                 ("REALISTIX_HELDOUT_CELLS", cells(fold))):
        os.environ[k] = v
    paths = sorted(glob.glob(os.path.join(run_glob, "best.pt")))
    if not paths:
        raise SystemExit(f"no checkpoints under {run_glob}")
    ens = Ensemble(paths, device)
    ds = build("real", "val_aoa", stride=40)
    P, Y = [], []
    for x, y, scale in DataLoader(ds, batch_size=32):
        x, y = x.to(device), y.to(device)
        sc = scale.to(device)[:, None, None, None]
        with torch.autocast("cuda", dtype=amp_dtype_for(device), enabled=device == "cuda"):
            p, _ = ens.mean_std(x)
        n = x.shape[0]
        shape = (n, T, 2, x.shape[-2], x.shape[-1])
        P.append((p.float() * sc).reshape(shape).cpu())
        Y.append((y * sc).reshape(shape).cpu())
    return torch.cat(P), torch.cat(Y)


def _kgrid(h: int, w: int):
    ky = (torch.fft.fftfreq(h).abs() * h)[:, None]
    kx = (torch.fft.fftfreq(w).abs() * w)[None, :]
    return torch.sqrt(ky ** 2 + kx ** 2)


def analyze(p: torch.Tensor, y: torch.Tensor) -> dict:
    n, t, c, h, w = p.shape
    out = {}

    # Sanity columns straight from the scorer's own code.
    sub = subscores(to_ntchw(p.reshape(n, t * c, h, w)).numpy(),
                    to_ntchw(y.reshape(n, t * c, h, w)).numpy())
    out.update({k: float(v) for k, v in sub.items()})

    pm, ym = p.mean(dim=1), y.mean(dim=1)                  # (N, C, H, W)
    pf, yf = p - pm[:, None], y - ym[:, None]              # fluctuation about own mean
    em = pm - ym
    ef = pf - yf

    # Exact orthogonal split of the MSE over t.
    out["mse_total"] = float((p - y).pow(2).mean())
    out["mse_mean"] = float(em.pow(2).mean())
    out["mse_fluct"] = float(ef.pow(2).mean())

    k = _kgrid(h, w)
    lo, mid = k < 2, (k >= 2) & (k < 8)

    def band2(x):                                          # (..., H, W) -> 3 energies
        E = torch.fft.fft2(x, norm="ortho").abs().pow(2)
        return [float(E[..., m].sum()) for m in (lo, mid, ~(lo | mid))]

    out["em_k"] = band2(em)

    # Fluctuation error in (omega, k). omega = 0 of ef is exactly zero.
    Ef = torch.fft.fftn(ef, dim=(1, 3, 4), norm="ortho").abs().pow(2)
    om = (torch.fft.fftfreq(t).abs() * t)[None, :, None, None, None]
    kk = k[None, None, None]
    slow = (om >= 1) & (om <= 2)
    k8 = kk < 8
    out["ef_bands"] = {
        "w12_k08": float(Ef[(slow & k8).expand_as(Ef)].sum()),
        "w12_k8p": float(Ef[(slow & ~k8).expand_as(Ef)].sum()),
        "w3p_k08": float(Ef[(~slow & k8).expand_as(Ef)].sum()),
        "w3p_k8p": float(Ef[(~slow & ~k8).expand_as(Ef)].sum()),
    }

    # Where the amplitude sits: predicted against target fluctuation energy by k.
    Ep = torch.fft.fftn(pf, dim=(1, 3, 4), norm="ortho").abs().pow(2)
    Ey = torch.fft.fftn(yf, dim=(1, 3, 4), norm="ortho").abs().pow(2)
    out["amp_p_k"] = [float(Ep[kk.expand_as(Ep) < 2].sum()),
                      float(Ep[((kk >= 2) & (kk < 8)).expand_as(Ep)].sum()),
                      float(Ep[(kk >= 8).expand_as(Ep)].sum())]
    out["amp_y_k"] = [float(Ey[kk.expand_as(Ey) < 2].sum()),
                      float(Ey[((kk >= 2) & (kk < 8)).expand_as(Ey)].sum()),
                      float(Ey[(kk >= 8).expand_as(Ey)].sum())]

    # The bulk, counterfactually: which error component keeps elements out of a
    # fixed near-optimal band. Scored elements only, as the scorer does.
    m = y != 0
    cov = lambda e: float(((e.abs() <= H_BULK) & m).sum() / m.sum())
    out["cov_full"] = cov(p - y)
    out["cov_no_mean"] = cov(ef)                            # mean error removed
    out["cov_no_fluct"] = cov(em[:, None].expand_as(p))     # fluct error removed
    return out


def cache_path(arm: str, fold: str) -> str:
    """v2: the v1 files were all measured on the AoA=20 split (FINDINGS section 10)."""
    return os.path.join(RUNS, f"anatomy_v2_{arm}_{fold}.json")


def measure(arm: str, fold: str, device: str, refresh: bool) -> dict:
    cache = cache_path(arm, fold)
    if os.path.exists(cache) and not refresh:
        return json.load(open(cache))
    p, y = gather(os.path.join(RUNS, f"kf_{arm}_{fold}_s0"), fold, device)
    r = analyze(p, y)
    json.dump(r, open(cache, "w"), indent=1)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["mu02", "spsw05", "spsw20", "spsw80"])
    ap.add_argument("--ref", default="mu02")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--fold", default=None,
                    help="child mode: measure this one fold and cache it, then exit. "
                         "Set by the parent via foldenv.rerun_per_fold; the holdout "
                         "must already be in the environment when this process starts.")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    folds = [f["name"] for f in MIMIC_FOLDS]

    if args.fold is not None:                      # child: one fold, then exit
        for a in args.arms:
            measure(a, args.fold, device, args.refresh)
        return

    missing = [(a, f) for a in args.arms for f in folds
               if args.refresh or not os.path.exists(cache_path(a, f))]
    if missing:
        foldenv.rerun_per_fold(sorted({f for _, f in missing}),
                               base_argv=["--arms", *args.arms]
                                         + (["--refresh"] if args.refresh else []))

    # Parent reads caches only: it imported splits under no holdout, so anything it
    # measured itself would be the AoA=20 split again.
    R = {}
    for a in args.arms:
        for f in folds:
            p = cache_path(a, f)
            if not os.path.exists(p):
                raise SystemExit(f"missing cache {p} -- a child fold failed")
            R[(a, f)] = json.load(open(p))

    seen = {tuple(v["amp_y_k"]) for v in R.values()}
    if len(seen) == 1 and len(R) > 1:
        raise SystemExit(
            "every cache carries the same target amplitude vector, so the fold did not "
            "take effect -- see FINDINGS.md section 10. Delete the caches and re-run.")

    def dmean(arm, key, sub=None, idx=None):
        vals = []
        for f in folds:
            a, b = R[(arm, f)], R[(args.ref, f)]
            va = a[key][sub] if sub else (a[key][idx] if idx is not None else a[key])
            vb = b[key][sub] if sub else (b[key][idx] if idx is not None else b[key])
            vals.append(va - vb)
        return statistics.fmean(vals)

    def bmean(key, sub=None, idx=None):
        vals = []
        for f in folds:
            b = R[(args.ref, f)]
            vals.append(b[key][sub] if sub else (b[key][idx] if idx is not None else b[key]))
        return statistics.fmean(vals)

    def sse_ref(f):
        b = R[(args.ref, f)]
        return T * sum(b["em_k"]) + sum(b["ef_bands"].values())

    def dband(arm, key, sub=None, idx=None, scale=1.0):
        """Band delta as a fraction of the reference fold's TOTAL squared error."""
        vals = []
        for f in folds:
            a, b = R[(arm, f)], R[(args.ref, f)]
            va = a[key][sub] if sub else a[key][idx]
            vb = b[key][sub] if sub else b[key][idx]
            vals.append(scale * (va - vb) / sse_ref(f))
        return statistics.fmean(vals)

    mt = bmean("mse_total")
    print(f"reference {args.ref}: mse_mean {100*bmean('mse_mean')/mt:.1f}% of total, "
          f"mse_fluct {100*bmean('mse_fluct')/mt:.1f}%   (4 folds, seed 0, val_aoa holdout)")
    print(f"fluct amplitude ratio pred/target by k:  "
          + "  ".join(f"{b} {bmean('amp_p_k', idx=i)/bmean('amp_y_k', idx=i):.3f}"
                      for i, b in enumerate(("k<2", "2-8", ">=8"))))

    print(f"\n{'arm':8} {'d rel_l2':>9} {'d tke':>7} | {'d mse_mean':>10} {'d mse_flct':>10} | "
          f"{'d em k<2':>9} {'2-8':>7} {'>=8':>7} | {'d ef w12k8':>10} {'w3p_k08':>8}")
    print("-" * 106)
    for arm in args.arms:
        if arm == args.ref:
            continue
        print(f"{arm:8} {dmean(arm, 'rel_l2_score'):+9.3f} {dmean(arm, 'tke_score'):+7.3f} | "
              f"{100*dmean(arm, 'mse_mean')/mt:+9.2f}% {100*dmean(arm, 'mse_fluct')/mt:+9.2f}% | "
              f"{100*dband(arm, 'em_k', idx=0, scale=T):+8.2f}% "
              f"{100*dband(arm, 'em_k', idx=1, scale=T):+6.2f}% "
              f"{100*dband(arm, 'em_k', idx=2, scale=T):+6.2f}% | "
              f"{100*dband(arm, 'ef_bands', sub='w12_k08'):+9.2f}% "
              f"{100*dband(arm, 'ef_bands', sub='w3p_k08'):+7.2f}%")

    print(f"\nfluct amplitude pred/target, delta by k band (negative = MORE damping):")
    for arm in args.arms:
        if arm == args.ref:
            continue
        cells_ = []
        for i, b in enumerate(("k<2", "2-8", ">=8")):
            r1 = statistics.fmean(R[(arm, f)]["amp_p_k"][i] / R[(arm, f)]["amp_y_k"][i]
                                  for f in folds)
            r0 = statistics.fmean(R[(args.ref, f)]["amp_p_k"][i] / R[(args.ref, f)]["amp_y_k"][i]
                                  for f in folds)
            cells_.append(f"{b} {r1 - r0:+.4f}")
        print(f"  {arm:8} " + "   ".join(cells_))

    print(f"\nbulk coverage at h = 0.124 sigma (counterfactual split):")
    print(f"  {'arm':8} {'full':>8} {'no mean err':>12} {'no fluct err':>13}   "
          f"(delta vs {args.ref})")
    for arm in args.arms:
        if arm == args.ref:
            continue
        print(f"  {arm:8} {100*dmean(arm, 'cov_full'):+8.3f} "
              f"{100*dmean(arm, 'cov_no_mean'):+12.3f} {100*dmean(arm, 'cov_no_fluct'):+13.3f}")
    print(f"\n  reference levels: full {100*bmean('cov_full'):.2f}%  "
          f"no-mean {100*bmean('cov_no_mean'):.2f}%  no-fluct {100*bmean('cov_no_fluct'):.2f}%")
    print("\n  read: if the arm's 'full' gain reappears in 'no fluct err' (the mean-only")
    print("  error) the bulk gain lives in the time-mean and is separable from TKE;")
    print("  if it reappears only in 'no mean err' it IS the fluctuation damping.")


if __name__ == "__main__":
    main()
