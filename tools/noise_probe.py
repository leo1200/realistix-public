#!/usr/bin/env python3
"""Does injecting the measured PIV noise floor into predictions pay?

FINDINGS 9b: the measured TKE the score compares against includes the noise
variance sigma_n^2, which a smooth forecast cannot carry — buying it with
coherent amplitude costs rel_l2 quadratically (the w_tke dose curve already
sits at that trade's optimum). Incoherent white noise at inference adds the
same variance at the minimal rel cost, and leaves the head's intervals — and
so the SPS width factor — untouched. This scores that rider on the ingrid rig
at several noise scales lam * sigma_n (per channel, the 9b global means).

    python tools/noise_probe.py submissions/_honest_onelean/submission.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(HERE, "..", "starting_kit")
SIGMA_N = (0.00733, 0.00147)   # u, v: FINDINGS 9b phase-residual estimates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--eval-dir", default="eval/ingrid")
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.submission)))
    import importlib.util
    spec = importlib.util.spec_from_file_location("sub_probe", args.submission)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    predict = (mod.predict if hasattr(mod, "predict")
               else mod.SubmissionModel(submission_dir=os.path.dirname(
                   os.path.abspath(args.submission))).predict)

    x = np.load(os.path.join(args.eval_dir, "samples.npz"))["input"]
    out = predict(x, {})
    base = dict(out) if isinstance(out, dict) else {"prediction": np.asarray(out)}
    base = {k: np.asarray(v, np.float32) for k, v in base.items()}
    rng = np.random.default_rng(args.seed)
    noise = rng.standard_normal(base["prediction"][..., :2].shape).astype(np.float32)
    noise *= np.array(SIGMA_N, np.float32)

    for lam in args.lams:
        bundle = dict(base)
        p = bundle["prediction"].copy()
        p[..., :2] += lam * noise
        bundle["prediction"] = p
        bundle["mean_t_neural_s"] = np.array([1.33e-3], np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            np.savez_compressed(os.path.join(tmp, "predictions.npz"), **bundle)
            os.symlink(os.path.abspath(os.path.join(args.eval_dir, "ref")),
                       os.path.join(tmp, "ref"))
            subprocess.run([sys.executable, os.path.join(KIT, "scoring.py"), tmp, tmp],
                           check=True, capture_output=True)
            sc = json.load(open(os.path.join(tmp, "scores.json")))
        five = [v for k, v in sc.items() if k.endswith("_score")]
        keys = ["rel_l2_score", "tke_score", "mvpe_score", "sps_score"]
        line = "  ".join(f"{k.split('_score')[0]}={sc[k]:7.3f}" for k in keys)
        print(f"lam={lam:4.2f}  {line}  mean5={np.mean(five):7.3f}", flush=True)


if __name__ == "__main__":
    main()
