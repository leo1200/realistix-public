#!/usr/bin/env python3
"""Cut a local evaluation set in the Codabench layout.

    <out>/samples.npz        input  (N, 20, 32, 64, 3)
    <out>/ref/targets.npz    target (N, 20, 32, 64, 3)

so ``starting_kit/scoring.py <out> <out>`` scores it exactly as the leaderboard
would. Windows are cut at stride >= T_in + T_out, so no window's target lands in
another window's input -- the leak that forced the 2026-08-05 phase restart.

    python tools/build_eval_npz.py val_re eval/val_re
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from realistix.data import frame_range  # noqa: E402
from realistix.splits import partition  # noqa: E402

T_IN = T_OUT = 20
SUB = 2


def windows(traj_paths, stride, max_per_traj=None, frames="all"):
    ins, tgts, ids = [], [], []
    for path in traj_paths:
        with h5py.File(path, "r") as h:
            u = h["u"][:, ::SUB, ::SUB].astype(np.float32)
            v = h["v"][:, ::SUB, ::SUB].astype(np.float32)
        field = np.stack([u, v, np.zeros_like(u)], axis=-1)
        stem = os.path.basename(path)[:-3]
        n = 0
        lo, hi = frame_range(field.shape[0], frames)
        for s in range(lo, hi - (T_IN + T_OUT) + 1, stride):
            if max_per_traj is not None and n >= max_per_traj:
                break
            ins.append(field[s:s + T_IN])
            tgts.append(field[s + T_IN:s + T_IN + T_OUT])
            ids.append(f"{stem}_{s}")
            n += 1
    return np.stack(ins), np.stack(tgts), np.array(ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["train", "val_re", "val_aoa", "test_aoa"])
    ap.add_argument("--frames", default="all", choices=["all", "head", "tail"],
                    help="with REALISTIX_HELDOUT_TAIL set, 'tail' cuts the held-out "
                         "segment of every trajectory -- the honest in-grid eval set")
    ap.add_argument("out")
    ap.add_argument("--stride", type=int, default=T_IN + T_OUT)
    ap.add_argument("--max-per-traj", type=int, default=None)
    args = ap.parse_args()

    if args.stride < T_IN + T_OUT:
        raise SystemExit(f"stride {args.stride} < {T_IN + T_OUT} would leak targets across windows")

    trajs = partition("real")[args.split]
    x, y, ids = windows([t.path("real") for t in trajs], args.stride,
                        args.max_per_traj, args.frames)

    os.makedirs(os.path.join(args.out, "ref"), exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "samples.npz"), input=x, sample_id=ids)
    np.savez_compressed(os.path.join(args.out, "ref", "targets.npz"), target=y)
    print(f"{args.split}: {len(trajs)} trajectories -> {x.shape[0]} windows {x.shape[1:]}")
    print(f"  masked (exact-zero) fraction of measured targets: {(y[..., :2] == 0).mean():.4f}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
