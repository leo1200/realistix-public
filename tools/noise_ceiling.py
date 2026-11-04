#!/usr/bin/env python3
"""How much of the remaining error is plausibly PIV noise? Score a denoising oracle.

**"Ceiling" is the wrong word and this file used to use it** (fixed 2026-09-01).
What is computed is the score of one particular band-limited reconstruction under
one noise model, given perfect forecasting of the denoised field. That is not a
supremum over predictors: the naive prefilter below already *beats* the "Wiener
ceiling" on TKE, which a genuine bound could not allow. Use these numbers to
estimate how much of the residual is target noise; do not use them to close an axis.

The hidden targets are measured fields and carry per-vector PIV noise that no
model can predict. This tool scores *oracles* against the same noisy targets the
honest gates use (the 237-window ingrid rig), through the kit's own scoring.py,
so the reference point lands in the same units as every gate read:

  identity   the noisy phase-0 target itself     -> 100 everywhere (rig sanity)
  wiener     per-trajectory Wiener filter of the raw 64x128 fluctuations, noise
             PSD measured from the decimation-phase residuals -> the best
             band-limited estimate of the field under this noise model; its
             score against the noisy target is a reference point, NOT a supremum
             -- `phasemean` below beats it on tke, which a real bound forbids
  phasemean  a centered [1,2,1]^2 prefilter on the raw grid -- the naive
             denoiser, for contrast with wiener

Bounds for the SPS column are the per-element optimum given the oracle's own
error: the tightest interval containing the target, reward exp(-|e|/sigma). That
reads the target, so it is an oracle reading rather than anything achievable -- and
because it is conditioned on this particular denoiser, it is not an upper bound on
what some other predictor could score either.

    python tools/noise_ceiling.py                       # all three oracles
    python tools/noise_ceiling.py --oracles wiener
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(HERE, "..", "starting_kit")
DATA = os.environ.get(
    "REALPDE_DATA",
    "data/RealPDE-Competition-Data/training_data")


def noise_sigma(raw: np.ndarray) -> float:
    """Per-vector noise RMS of one channel, from the decimation-phase residual.

    The diagonal phase against the bilinear interpolation of its four kept
    neighbours: for iid noise the residual variance is sigma^2 * (1 + 4/16);
    the signal part is the interpolation error of a field with 0.2% of its
    energy above the coarse Nyquist, i.e. noise dominates (measured 2026-08-31:
    the residual RMS is constant across regimes while fluctuation RMS varies 3x).
    """
    est = 0.25 * (raw[:, :-2:2, :-2:2] + raw[:, :-2:2, 2::2]
                  + raw[:, 2::2, :-2:2] + raw[:, 2::2, 2::2])
    resid = raw[:, 1:-1:2, 1:-1:2] - est
    return float(np.sqrt(resid.var() / 1.25))


def denoise(raw: np.ndarray, kind: str) -> np.ndarray:
    """(T, 64, 128) raw channel -> (T, 32, 64) denoised estimate on the eval grid."""
    if kind == "identity":
        return raw[:, ::2, ::2]
    if kind == "phasemean":
        k = np.array([1.0, 2.0, 1.0]) / 4.0
        sm = raw.copy()
        for ax, dim in ((1, 64), (2, 128)):
            pad = np.take(sm, [0], axis=ax), np.take(sm, [dim - 1], axis=ax)
            sm = np.concatenate([pad[0], sm, pad[1]], axis=ax)
            sm = sum(w * np.take(sm, np.arange(i, i + dim), axis=ax)
                     for i, w in enumerate(k))
        return sm[:, ::2, ::2]
    assert kind == "wiener"
    mean = raw.mean(0)
    fluct = raw - mean
    F = np.fft.fft2(fluct)
    p_meas = (np.abs(F) ** 2).mean(0)
    p_noise = noise_sigma(raw) ** 2 * raw.shape[1] * raw.shape[2]
    gain = np.clip(1.0 - p_noise / np.maximum(p_meas, 1e-30), 0.0, 1.0)
    return (mean + np.fft.ifft2(F * gain).real)[:, ::2, ::2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="eval/ingrid")
    ap.add_argument("--oracles", nargs="+", default=["identity", "wiener", "phasemean"])
    ap.add_argument("--time-ms", type=float, default=1.33,
                    help="mean_t_neural_s to report, so 'mean of five' is "
                         "comparable to the honest gates (onelean: 1.33)")
    args = ap.parse_args()

    ids = np.load(os.path.join(args.eval_dir, "samples.npz"))["sample_id"]
    target = np.load(os.path.join(args.eval_dir, "ref", "targets.npz"))["target"]
    windows = [(i, *s.rsplit("_", 1)) for i, s in enumerate(ids)]
    stems = sorted({stem for _, stem, _ in windows})

    for kind in args.oracles:
        pred = np.zeros_like(target)
        sigmas = []
        for stem in stems:
            with h5py.File(os.path.join(DATA, "train_real", stem + ".h5"), "r") as h:
                u, v = h["u"][:].astype(np.float64), h["v"][:].astype(np.float64)
            sigmas.append((noise_sigma(u), noise_sigma(v)))
            du, dv = denoise(u, kind), denoise(v, kind)
            for i, s, start in windows:
                if s != stem:
                    continue
                t0 = int(start) + 20
                pred[i, :, :, :, 0] = du[t0:t0 + 20]
                pred[i, :, :, :, 1] = dv[t0:t0 + 20]
        # Oracle SPS bounds: the tightest interval containing the target. Reads
        # the target, so it is a reference reading and not achievable.
        lower = np.minimum(pred, target).astype(np.float32)
        upper = np.maximum(pred, target).astype(np.float32)
        bundle = dict(prediction=pred.astype(np.float32), lower=lower, upper=upper,
                      mean_t_neural_s=np.array([args.time_ms / 1e3], np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            np.savez_compressed(os.path.join(tmp, "predictions.npz"), **bundle)
            os.symlink(os.path.abspath(os.path.join(args.eval_dir, "ref")),
                       os.path.join(tmp, "ref"))
            subprocess.run([sys.executable, os.path.join(KIT, "scoring.py"), tmp, tmp],
                           check=True, capture_output=True)
            sc = json.load(open(os.path.join(tmp, "scores.json")))
        su = np.array([s[0] for s in sigmas]); sv = np.array([s[1] for s in sigmas])
        print(f"\n=== oracle: {kind}  |  {len(ids)} windows  |  optimal bounds ===")
        print(f"  sigma_n u: {su.mean():.5f} (range {su.min():.5f}-{su.max():.5f})"
              f"   v: {sv.mean():.5f}")
        for k, val in sc.items():
            if isinstance(val, float):
                print(f"  {k:<16}{val:8.3f}")
        five = [sc[k] for k in sc if k.endswith("_score")]
        print(f"  {'mean of five':<16}{np.mean(five):8.3f}")


if __name__ == "__main__":
    main()
