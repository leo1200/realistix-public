"""RealPDEBench foil (HF Arrow, 128x256 at dt 0.005) -> our release layout (64x128 at dt 0.02).

Every 4th frame; 2x2 block mean in space (a block with any exact zero stays zero, so
masked cells stay masked); rows flipped so row 0 is the top, as in our files. Writes
conv/{real,sim}/<re>_<aoa>.h5 with top-level u, v (and p for sim), t, re, aoa.
The trainer's loader then takes [::2, ::2] to reach the 32x64 evaluation grid.
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = os.path.dirname(os.path.abspath(__file__))
TSUB = 4


def coarse(a):
    a = a[::TSUB]
    T, H, W = a.shape
    b = a.reshape(T, H // 2, 2, W // 2, 2)
    m = b.mean(axis=(2, 4))
    m[(b == 0).any(axis=(2, 4))] = 0.0
    return m[:, ::-1].astype(np.float32)


def one(job):
    path, kind, params = job
    t = ipc.open_stream(pa.memory_map(path)).read_all()
    out = []
    for i in range(t.num_rows):
        g = lambda c: t.column(c)[i].as_py()
        sid = g("sim_id")
        re, aoa = params[sid]
        T, H, W = g("shape_t"), g("shape_h"), g("shape_w")
        dst = os.path.join(ROOT, "conv", kind, f"{int(re)}_{int(aoa)}.h5")
        with h5py.File(dst + ".tmp", "w") as h:
            for c in ("u", "v") + (("p",) if kind == "sim" else ()):
                h[c] = coarse(np.frombuffer(g(c), np.float32).reshape(T, H, W))
            h["t"] = np.frombuffer(g("t"), np.float64)[::TSUB]
            h["re"], h["aoa"] = float(re), float(aoa)
            h.attrs["source"] = f"RealPDEBench 2.0.1 foil/{kind} {sid}"
        os.replace(dst + ".tmp", dst)
        out.append(dst)
    return out


if __name__ == "__main__":
    jobs = []
    for kind, src in (("real", "real"), ("sim", "numerical")):
        params = {}
        for part in ("remain", "in_dist_test", "out_dist_test"):
            params.update(json.load(open(f"{ROOT}/foil/{part}_params_{src}.json")))
        os.makedirs(f"{ROOT}/conv/{kind}", exist_ok=True)
        jobs += [(p, kind, params) for p in sorted(glob.glob(f"{ROOT}/foil/hf_dataset/{src}/*.arrow"))]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        for r in pool.imap_unordered(one, jobs):
            print(*r, flush=True)
    print("DONE", len(jobs))
