#!/usr/bin/env python3
"""Can a network turn a simulated window into a real-looking one?

A diagnostic, not a component. Its output must never become training data for the
shipped model -- that is the line the competition rules draw -- and it is deliberately
absent from ``tools/train_final.sh``. The question it answers is whether the sim->real
map is learnable at all, which is the one open question about this track's premise.

**There is no paired supervision.** Sim and real trajectories at the same (Re, AoA)
are different realisations of the same statistical flow: their time alignment
correlates 0.02 at every lag. A pointwise loss between them is meaningless. What is
paired is the *regime* -- 81 usable ones, 57 in the default training split -- so this
is distribution translation with the regime as the pairing key. (The counts said 82
and 58 until 2026-09-01, from before `7575_0` was withdrawn.)

**Parameterised as a correction, not a resynthesis.** Across those regimes the
simulation already tracks the real variation (corr 0.63-0.90 on fluctuation level,
wake width, deficit depth, cross-flow) while its levels are wrong by different
factors per quantity (3.42x, 2.35x, 1.49x, 1.06x). Structure is right and magnitude
is wrong, so the generator outputs ``x + delta`` from a zero-initialised conv: it
starts as the identity and can only learn the correction. That also removes any need
for a structure-preservation term, which would otherwise fight the objective, since
the wake is *supposed* to get 1.5x narrower.

**Success is pre-registered.** Trained on the 58 `train` regimes and evaluated on the
24 held out, the four ratios above should move toward 1.0 while the cross-regime
correlations survive. Matching statistics on regimes it was fitted to would only show
memorisation; whether the *map* generalises is the entire question.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from realistix.data import T_IN, frame_range, load_traj  # noqa: E402
from realistix.models.unet import UNet, count_params  # noqa: E402
from realistix.normalize import EPS  # noqa: E402
from realistix.splits import partition  # noqa: E402

T, C = T_IN, 2


class RegimePairs:
    """Real and simulated frames for each paired regime, resident on the GPU."""

    def __init__(self, trajs, device, frames="all"):
        self.device, self.keys, self.real, self.sim = device, [], {}, {}
        self.frames, skipped = frames, []
        for t in trajs:
            try:
                r = torch.from_numpy(load_traj(t.path("real"))).to(device)
                s = torch.from_numpy(load_traj(t.path("sim"))).to(device)
            except FileNotFoundError as e:
                # Only a genuinely absent pair is a skip. `except Exception` here hid
                # corrupt HDF5, a wrong root and CUDA OOM behind the same silent
                # `continue`, so an experiment could quietly lose regimes and carry on
                # with a smaller, biased population. Note OSError is deliberately NOT
                # caught: h5py raises it for a truncated or corrupt file, which is a
                # data problem to fix, not a regime that happens to be unpaired.
                skipped.append(f"{t.stem}: {type(e).__name__}")
                continue
            self.keys.append(t.stem)
            self.real[t.stem], self.sim[t.stem] = r, s
        if skipped:
            print(f"RegimePairs: {len(skipped)} unpaired regime(s) skipped: "
                  + "; ".join(skipped), flush=True)
        if not self.keys:
            raise SystemExit("RegimePairs found no paired regimes; check the data root")

    def _draw(self, field, batch):
        # Honour the tail holdout: this sampler drew from the full frame range even
        # under REALISTIX_HELDOUT_TAIL, so `--aux-s2r` pulled statistics out of the
        # very tail the run was being evaluated on.
        lo, hi = frame_range(field.shape[0], self.frames)
        field = field[lo:hi]
        n = field.shape[0] - T
        i = torch.randint(0, n, (batch,), device=self.device)
        w = torch.arange(T, device=self.device)
        out = field[i[:, None] + w[None, :]]                    # (B, T, 2, H, W)
        # Per-window normalisation, exactly as the forecaster sees it: this is what
        # divides out the m/s versus non-dimensional unit gap, leaving only the
        # differences that are actually physical.
        tot = out.pow(2).sum(dim=(1, 2, 3, 4))
        nz = (out.abs() >= EPS).any(dim=2).any(dim=1)
        sc = (tot / (T * nz.sum(dim=(1, 2)).clamp_min(1))).sqrt() + EPS
        out = out / sc[:, None, None, None, None]
        return out.reshape(batch, T * C, *out.shape[-2:])

    def batch(self, stem, batch):
        return self._draw(self.sim[stem], batch), self._draw(self.real[stem], batch)


def stats(w):
    """Statistics a translation has to get right, all batch-averaged."""
    f = w.reshape(w.shape[0], T, C, *w.shape[-2:])
    mean = f.mean(dim=1)                                   # (B, C, H, W) time-mean
    tke = 0.5 * ((f - mean[:, None]) ** 2).mean(dim=1).sum(dim=1)   # (B, H, W)
    spec = torch.fft.rfft2(f - mean[:, None], norm="ortho").abs().mean(dim=(1, 2))
    return mean.mean(0), tke.mean(0), spec.mean(0)


def match_loss(a, b):
    return sum(torch.nn.functional.mse_loss(x, y) for x, y in zip(a, b))


@torch.no_grad()
def ratios(gen, data, stems, device, batch=32, reps=4):
    """sim/real and translated/real fluctuation-level ratio on given regimes.

    Narrower than the four ratios the header motivates: this reports statistic index
    1 (mean TKE / fluctuation level) only. Wake width, deficit depth and cross-flow
    are computed by `stats` and never reduced here, and the cross-regime correlation
    the header calls a preservation target is not evaluated at all. Read a passing
    number as "the fluctuation level was corrected", not as "the translation
    preserves structure".
    """
    acc = {k: [] for k in ("raw", "gen")}
    for st in stems:
        rs, gs, ts = [], [], []
        for _ in range(reps):
            s, r = data.batch(st, batch)
            g = s + gen(s) if gen is not None else s
            rs.append([x.float() for x in stats(s)])
            gs.append([x.float() for x in stats(g)])
            ts.append([x.float() for x in stats(r)])
        def red(v, i):
            return float(torch.stack([q[i] for q in v]).mean())
        # index 1 is the TKE field; its spatial mean is the fluctuation level
        acc["raw"].append(red(rs, 1) / max(red(ts, 1), 1e-9))
        acc["gen"].append(red(gs, 1) / max(red(ts, 1), 1e-9))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = partition("real")
    train = RegimePairs(p["train"], device)
    held = RegimePairs(p["val_re"] + p["val_aoa"] + p["test_aoa"], device)
    print(f"regimes: {len(train.keys)} for training, {len(held.keys)} held out",
          flush=True)

    gen = UNet(width=args.width, n_modalities=0, predict_delta=False).to(device)
    print(f"generator: {count_params(gen)/1e6:.2f}M params", flush=True)
    opt = torch.optim.AdamW(gen.parameters(), lr=args.lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    base_tr = ratios(None, train, train.keys, device)
    base_ho = ratios(None, held, held.keys, device)
    print(f"before training: fluctuation ratio sim/real = {base_tr['raw']:.2f} "
          f"(train regimes), {base_ho['raw']:.2f} (held out)\n", flush=True)

    hist, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        st = train.keys[torch.randint(len(train.keys), (1,)).item()]
        s, r = train.batch(st, args.batch)
        loss = match_loss(stats(s + gen(s)), stats(r))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt.step(); sch.step()
        if step % args.eval_every == 0:
            a, b = ratios(gen, train, train.keys, device), ratios(gen, held, held.keys, device)
            print(f"[{step:>5}] loss {float(loss):.5f}  fluctuation ratio -> "
                  f"train {a['gen']:.2f}  held-out {b['gen']:.2f}   "
                  f"(target 1.00, untranslated {b['raw']:.2f})  [{time.time()-t0:.0f}s]",
                  flush=True)
            hist.append({"step": step, "train": a, "held_out": b})
            json.dump(hist, open(os.path.join(args.out, "history.json"), "w"), indent=1)
            torch.save({"gen": gen.state_dict(), "args": vars(args)},
                       os.path.join(args.out, "gen.pt"))


if __name__ == "__main__":
    main()
