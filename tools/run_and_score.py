#!/usr/bin/env python3
"""Run a submission over a local eval set and score it with the kit's scoring.py.

    python tools/run_and_score.py submissions/fno_band/submission.py \
        eval/val_re

Mimics the real ingestion program in the ways that can bite: predict() is called
in chunks (the scored run calls it more than once) with an empty metadata dict,
and the timed region excludes a warm-up pass, matching the note that checkpoint
loading is not counted toward the Time metric.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

KIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "starting_kit")


def load_predict(path):
    spec = importlib.util.spec_from_file_location("submission_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)
    if hasattr(mod, "predict"):
        return mod.predict
    model = mod.SubmissionModel(submission_dir=os.path.dirname(os.path.abspath(path)))
    return model.predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("eval_dir")
    ap.add_argument("--chunk", type=int, default=64, help="windows per predict() call")
    args = ap.parse_args()

    wall0 = time.perf_counter()
    predict = load_predict(args.submission)
    x = np.load(os.path.join(args.eval_dir, "samples.npz"))["input"]

    predict(x[:4], {})  # warm-up; loading is not part of the Time metric

    chunks, elapsed = [], 0.0
    for i in range(0, x.shape[0], args.chunk):
        t0 = time.perf_counter()
        chunks.append(predict(x[i:i + args.chunk], {}))
        elapsed += time.perf_counter() - t0
    # There is a wall-clock limit on the prediction stage, independent of the Time
    # metric, and a submission that exceeds it is marked Failed rather than scored.
    # Loading counts toward it even though it does not count toward Time. Track 1's
    # binding figure is 180 s (competition_info/hardware.md, 2026-08-27); the FAQ's
    # 5 minutes is the looser platform-wide number.
    wall = time.perf_counter() - wall0

    # "Bounds are all-or-nothing across the whole run: predict may be called more
    # than once, and if some calls return bounds and others do not, every bound is
    # discarded and the default band is used instead." This used to take the key set
    # from the first chunk and index every later one with it, which raises a KeyError
    # on exactly the mixed-availability case the rule exists for -- so the fallback
    # the leaderboard would apply was never exercised locally.
    if isinstance(chunks[0], dict):
        keys = set(chunks[0])
        for c in chunks[1:]:
            keys &= set(c)
        for b in ("lower", "upper"):
            if b not in keys and any(b in c for c in chunks):
                print(f"note: {b!r} missing from some predict() calls -- the official "
                      f"scorer discards ALL bounds here and uses the default band")
        if not {"lower", "upper"} <= keys:
            keys -= {"lower", "upper"}
        bundle = {k: np.concatenate([np.asarray(c[k], np.float32) for c in chunks])
                  for k in sorted(keys)}
    else:
        bundle = {"prediction": np.concatenate([np.asarray(c, np.float32) for c in chunks])}
    bundle["mean_t_neural_s"] = np.array([elapsed / x.shape[0]], np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        np.savez_compressed(os.path.join(tmp, "predictions.npz"), **bundle)
        os.symlink(os.path.abspath(os.path.join(args.eval_dir, "ref")), os.path.join(tmp, "ref"))
        subprocess.run([sys.executable, os.path.join(KIT, "scoring.py"), tmp, tmp], check=True)
        scores = json.load(open(os.path.join(tmp, "scores.json")))

    bounds = "with bounds" if "lower" in bundle else "no bounds (default band)"
    print(f"\n{os.path.basename(os.path.dirname(args.submission))} on "
          f"{os.path.basename(args.eval_dir)}  |  {x.shape[0]} windows  |  {bounds}")
    print(f"  {elapsed / x.shape[0] * 1000:.2f} ms/sample")
    WALL_CAP = float(os.environ.get("REALISTIX_WALL_CAP", 180.0))
    flag = "  <- OVER the Track 1 execution limit; this run would be Failed" \
        if wall > WALL_CAP else ""
    print(f"  {wall:.1f}s wall clock of {WALL_CAP:.0f}s cap{flag}")
    for k, v in scores.items():
        print(f"  {k:<16}{v:8.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  {'mean of five':<16}{np.mean([scores[k] for k in scores if k.endswith('_score')]):8.3f}")


if __name__ == "__main__":
    main()
