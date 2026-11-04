#!/usr/bin/env python3
"""Account for every millisecond inside predict(), split host from device.

The question this exists to answer has been asked three times and never answered.
`docs/archive/HANDOFF_2026-09-10.md` §4 records three submissions spent on `time_score`, each testing whether
some *change* moved it; none established what the timed region is made of. The
decisive fact is 891698: removing **76% of the parameters and 29% of the op sites**
moved the board from 8.261 to 8.225 ms/sample. So the network is under 2% of what the
container charges -- while a local profile puts it at 43-67% of `predict()`. Both
cannot describe the same execution, and nobody chased the contradiction.

What is timed is now pinned down. `scoring.py` reads `mean_t_neural_s` from the
prediction bundle, so the platform's ingestion harness measures it around the
`predict()` call; the FAQ confirms construction and checkpoint loading sit outside.
Everything else inside `predict()` is charged, network or not -- the input cast, the
host-to-device copies, the assembly of three full output tensors, the finiteness
guard, and three device-to-host transfers.

The split that matters is **host versus device**, because that is the axis on which
the container differs from this node. A weak-CPU container running a healthy GPU
would show exactly the observed signature: a network that is half the local profile
and none of the container's cost, with the remainder in numpy and transfers.

    python tools/time_breakdown.py --n 540 --repeat 3
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SUB = os.path.join(HERE, "..", "submissions", "all82_spread", "submission.py")


def load_submission(path):
    spec = importlib.util.spec_from_file_location("subm", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Clock:
    """Wall time per stage, with the device drained first so GPU work lands where
    it was actually issued rather than at the next synchronisation point."""

    def __init__(self, cuda: bool):
        self.cuda, self.t = cuda, {}
        self._last = None

    def __call__(self, name):
        if self.cuda:
            torch.cuda.synchronize()
        now = time.perf_counter()
        if self._last is not None:
            self.t[self._last[0]] = self.t.get(self._last[0], 0.0) + now - self._last[1]
        self._last = (name, now) if name else None


def instrumented(m, x, clk):
    """A transcription of predict(), stage by stage. Kept literal on purpose: if it
    drifts from the shipped function the numbers stop meaning anything."""
    T_IN, T_OUT, C = m.T_IN, m.T_OUT, m.C
    clk("input_cast")
    xf = np.asarray(x, dtype=np.float32)
    n, _, h, w, _ = xf.shape
    # _load grew a fourth return value (the combiner) on 2026-08-19. Unpacked
    # positionally so this stays readable against bundles from either side of that,
    # since the whole point of the transcription is that it matches the shipped
    # function -- an instrument that silently profiles a different code path is the
    # failure mode this repository has the most entries about.
    loaded = m._load()
    models, head, device = loaded[:3]
    combiner = loaded[3] if len(loaded) > 3 else None

    clk("alloc_out")
    out_dt = torch.float16 if device == "cuda" else torch.float32
    pred = torch.empty((n, T_OUT, h, w, 3), dtype=out_dt, device=device)
    lo, hi = torch.empty_like(pred), torch.empty_like(pred)

    with torch.no_grad():
        for i in range(0, n, m._BATCH):
            clk("h2d")
            xb = torch.from_numpy(xf[i:i + m._BATCH]).to(device, non_blocking=True)
            clk("preprocess")
            uv = xb[..., :2]
            scale = m._window_scale(uv)
            s = scale[:, None, None, None, None]
            xin = (uv / s).permute(0, 1, 4, 2, 3).reshape(-1, T_IN * C, h, w).contiguous()
            xin = xin.to(next(models[0].parameters()).dtype)
            clk("network")
            outs = torch.stack([mm(xin) for mm in models]).float()
            clk("combiner")
            y = outs.mean(0) if combiner is None else combiner(outs, xin.float())
            clk("spread")
            extra = None
            if head.extra_channels:
                r = outs.std(0, unbiased=True) * scale[:, None, None, None] / m.SIGMA_GLOBAL
                extra = torch.cat([r, torch.log(r.clamp_min(1e-4))], dim=1)
            clk("head")
            hdt = next(head.parameters()).dtype
            below, above = head(xin, y.to(hdt), torch.log(scale).to(hdt),
                                None if extra is None else extra.to(hdt))
            below, above = below.float(), above.float()
            clk("assemble")
            yy = y.reshape(-1, T_OUT, C, h, w).permute(0, 1, 3, 4, 2) * s
            bb = below.reshape(-1, T_OUT, C, h, w).permute(0, 1, 3, 4, 2)
            aa = above.reshape(-1, T_OUT, C, h, w).permute(0, 1, 3, 4, 2)
            sl = slice(i, i + m._BATCH)
            pred[sl, ..., :2] = yy.to(out_dt); pred[sl, ..., 2] = 0.0
            lo[sl, ..., :2] = (yy - bb).to(out_dt); lo[sl, ..., 2] = 0.0
            hi[sl, ..., :2] = (yy + aa).to(out_dt); hi[sl, ..., 2] = 0.0

    clk("guard")
    for t in (pred, lo, hi):
        torch.nan_to_num_(t, nan=0.0, posinf=0.0, neginf=0.0)
    torch.minimum(lo, pred, out=lo)
    torch.maximum(hi, pred, out=hi)
    clk("d2h_numpy")
    out = {"prediction": pred.cpu().numpy(), "lower": lo.cpu().numpy(),
           "upper": hi.cpu().numpy()}
    clk(None)
    return out


HOST = {"input_cast", "d2h_numpy", "h2d"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=DEFAULT_SUB)
    ap.add_argument("--n", type=int, default=540)
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()

    m = load_submission(os.path.abspath(a.sub))
    cuda = torch.cuda.is_available()
    src = np.load("eval/val_aoa/samples.npz")["input"]
    reps = int(np.ceil(a.n / len(src)))
    x = np.concatenate([src] * reps)[:a.n]
    print(f"{a.n} windows, batch {m._BATCH}, device "
          f"{'cuda' if cuda else 'cpu'}, {a.repeat} repeats after a warm-up\n")

    m.predict(x[:8])                                  # build the CUDA context first
    tot = {}
    for _ in range(a.repeat):
        clk = Clock(cuda)
        instrumented(m, x, clk)
        for k, v in clk.t.items():
            tot[k] = tot.get(k, 0.0) + v
    for k in tot:
        tot[k] /= a.repeat

    total = sum(tot.values())
    print(f"{'stage':13s} {'ms total':>9} {'ms/sample':>10} {'share':>7}   where")
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"{k:13s} {v*1e3:9.1f} {v*1e3/a.n:10.3f} {100*v/total:6.1f}%   "
              f"{'host' if k in HOST else 'device'}")
    host = sum(v for k, v in tot.items() if k in HOST)
    print(f"\n{'TOTAL':13s} {total*1e3:9.1f} {total*1e3/a.n:10.3f}")
    print(f"{'host-side':13s} {host*1e3:9.1f} {host*1e3/a.n:10.3f} {100*host/total:6.1f}%")
    print(f"{'device-side':13s} {(total-host)*1e3:9.1f} {(total-host)*1e3/a.n:10.3f} "
          f"{100*(total-host)/total:6.1f}%")
    net = tot.get("network", 0) + tot.get("head", 0)
    print(f"{'network+head':13s} {net*1e3:9.1f} {net*1e3/a.n:10.3f} {100*net/total:6.1f}%")
    print(f"\nThe board charges 8.261 ms/sample. 891698 removed 76% of the parameters")
    print(f"and moved it by 0.036 ms, so whatever dominates there is not the network.")
    print(f"Here the network is {100*net/total:.0f}% -- the gap between these two")
    print(f"statements is the whole question.")


if __name__ == "__main__":
    main()
