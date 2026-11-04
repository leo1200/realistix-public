#!/usr/bin/env python3
"""Re-clock the released simulation from convective to tunnel time (research/rb_2026-11-02 §3e).

The CFD runs at U = c = 1 for every Reynolds number with dt = 0.02 c/U; the PIV runs at dt = 0.02 s with the tunnel
speed setting Re. One measured frame therefore spans k = Re / Re0 simulated frames, and without correction a simulated
window evolves at 0.4-1.2x the measured pace per frame. Re0 = 4900 from two independent fits on cells away from 20 deg
(research/rb_2026-11-02/clock_fit.py): wake frequencies 4941, freestream speed with water's viscosity 4790 (chord
6.9 cm). Each trajectory is resampled at t_j = j * k simulated frames by linear interpolation in time (masked cells stay
zero). High-Re trajectories come out short (Re 27975: 175 frames) and low-Re ones long (Re 3750: 1306), so train with
--sim-equal-traj to keep the release's equal sampling mass per trajectory:

    python tools/clock_sim.py --dst data/train_sim_clk
    python tools/train_mixed.py ... --sim-dir data/train_sim_clk --sim-equal-traj

The output is a transform of released samples, not new data.
"""
import argparse
import glob
import os
from functools import partial
from multiprocessing import Pool

import h5py
import numpy as np


def one(path, dst_dir, re0):
    stem = os.path.basename(path)[:-3]
    re_, aoa = (int(v) for v in stem.split("_"))
    k = re_ / re0
    with h5py.File(path, "r") as h:
        u, v = h["u"][:], h["v"][:]
    T = u.shape[0]
    t = np.arange(0.0, T - 1 + 1e-9, k)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (t - i0)[:, None, None]
    dst = os.path.join(dst_dir, stem + ".h5")
    with h5py.File(dst + ".tmp", "w") as h:
        for name, a in (("u", u), ("v", v)):
            h.create_dataset(name, data=((1 - w) * a[i0] + w * a[i1]).astype(np.float32))
        h["re"], h["aoa"] = float(re_), float(aoa)
        h.attrs["k_sim_frames_per_real_frame"] = k
        h.attrs["source"] = f"train_sim/{stem}.h5 re-clocked with Re0 {re0:g}"
    os.replace(dst + ".tmp", dst)
    return stem, k, len(t)


def main() -> None:
    root = os.environ.get("REALPDE_DATA", "data/RealPDE-Competition-Data/training_data")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default=os.path.join(root, "train_sim"), help="released train_sim directory")
    ap.add_argument("--dst", required=True, help="output directory (pass to train_mixed.py --sim-dir)")
    ap.add_argument("--re0", type=float, default=4900.0, help="Reynolds number at which the two clocks agree")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    with Pool(a.workers) as pool:
        for stem, k, n in pool.imap_unordered(partial(one, dst_dir=a.dst, re0=a.re0),
                                              sorted(glob.glob(os.path.join(a.src, "*.h5")))):
            print(f"{stem} k={k:.3f} frames={n}", flush=True)
    with open(os.path.join(a.dst, "PROVENANCE.txt"), "w") as f:
        f.write(f"Released train_sim re-clocked to tunnel time: k = Re / {a.re0:g} simulated frames per measured\n"
                f"frame, linear interpolation in time. Tool: tools/clock_sim.py.\n")
    print("DONE")


if __name__ == "__main__":
    main()
