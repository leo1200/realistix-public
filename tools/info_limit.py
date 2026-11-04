#!/usr/bin/env python3
"""How much of the residual rel-L2 error is reachable, and how much is noise?

The forecaster sits at rel-L2 error 0.1044 on `val_aoa` and every configuration
knob is flat. That leaves one question unanswered, and it decides whether any
further work on the forecaster can pay: **is the remaining error information the
input window does not contain?** Nothing measured so far distinguishes "our model
is 5% short of the limit" from "our model is at the limit".

So measure the limit by cheating. Give the network, as extra input channels, part
of the *target* it is asked to predict, and see how far the error falls. A model
handed the target's large scales and still scoring error e sets a floor for a model
that *also* gets those large scales exactly right -- so the gap between our 0.1044
and that e prices **how much that component is worth**, and is reached only by a
model that predicts it perfectly.

It is not an upper bound on the headroom, which is how this read until 2026-09-01.
The oracle sees the target; a bound on what is inferable from the *inputs* is a
different quantity and is not measured anywhere in this repository.

Read it as a ladder:

* ``mean`` -- the target's time-averaged field. If this barely helps, our mean flow
  is already right and the error lives in the fluctuation, which is the part the
  coherence analysis says is unpredictable.
* ``k2 / k4 / k8`` -- the target's spatial content below that wavenumber, at every
  output frame. Predictability is concentrated at low k (coherence 0.90 at k in
  2-6 and t+1, 0.16 above k=12), so ``k4`` is roughly "the band where predictable
  structure lives".

  It is **not** "everything an honest model could hope to know", which is how this
  line read until 2026-09-01. The ladder hands the model part of its own *target*,
  so what it measures is how valuable perfect knowledge of a component would be --
  a floor on the error of a model *given* that component, not a bound on what can
  be inferred from the inputs alone. Nothing here measures an information limit
  (`docs/archive/FINDINGS_2026-09-10.md` §6).

**This is a diagnostic and its models must never ship.** It is deliberately
self-contained rather than an `--oracle` flag on `train_mixed.py`: a model that
reads the target has no legitimate path into a submission, and the cheapest way to
guarantee that is to keep it out of the production training script entirely. Same
line `tools/train_sim2real.py` sits on.

    python tools/info_limit.py --oracle none,mean,k4 --steps 4000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "starting_kit"))

import scoring as S  # noqa: E402
from realistix.data import build  # noqa: E402
from realistix.evaluate import subscores, to_ntchw  # noqa: E402
from realistix.gpudata import MixedGPUWindows  # noqa: E402
from realistix.losses import rel_l2, spectral, tke_rel_l2  # noqa: E402
from realistix.models.unet import UNet  # noqa: E402
from realistix.spectral import masks  # noqa: E402
from realistix.splits import partition  # noqa: E402

T_OUT, C = 20, 2


def oracle_channels(y, kind: str):
    """Extra input channels derived from the target ``y`` (B, T*C, H, W)."""
    if kind == "none":
        return None
    b, _, h, w = y.shape
    f = y.reshape(b, T_OUT, C, h, w)
    if kind == "mean":
        return f.mean(dim=1)                                   # (B, C, H, W)
    k = float(kind[1:])
    low, _ = masks(k, h, w, y.device)
    lo = torch.fft.irfft2(torch.fft.rfft2(f) * low, s=(h, w))
    return lo.reshape(b, T_OUT * C, h, w)


def n_extra(kind: str) -> int:
    return {"none": 0, "mean": C}.get(kind, T_OUT * C)


def loss_fn(pred, y):
    """The shipped objective: mse 1.0 + rel 8.0 + tke 0.5 + spec 0.1 + tmean 0.5."""
    f = lambda z: z.reshape(z.shape[0], T_OUT, C, *z.shape[-2:])
    tmean = torch.nn.functional.mse_loss(f(pred).mean(1), f(y).mean(1))
    return (torch.nn.functional.mse_loss(pred, y)
            + 8.0 * rel_l2(pred, y) + 0.5 * tke_rel_l2(pred, y)
            + 0.1 * spectral(pred, y) + 0.5 * tmean)


@torch.no_grad()
def evaluate(model, dl, device, kind):
    """Same de-normalisation as evaluate.predict_all, plus the oracle channels."""
    P, T = [], []
    model.eval()
    for x, y, scale in dl:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            o = oracle_channels(y, kind)
            xin = x if o is None else torch.cat([x, o], dim=1)
            out = model(xin, torch.zeros(x.shape[0], dtype=torch.long, device=device))
        out = out[0] if isinstance(out, tuple) else out
        s = scale.to(device)[:, None, None, None]
        P.append(to_ntchw(out.float() * s).cpu().numpy())
        T.append(to_ntchw(y.float() * s).cpu().numpy())
    model.train()
    return (np.concatenate(P).astype(np.float32), np.concatenate(T).astype(np.float32))


def run(kind, args, device):
    torch.manual_seed(args.seed)
    real = [t for t in partition("real")["train"]] + \
           ([] if not args.with_test_aoa else partition("real")["test_aoa"])
    sim = partition("sim")["train"] + partition("sim")["val_re"] + \
        partition("sim")["val_aoa"] + partition("sim")["test_aoa"]
    ds = MixedGPUWindows(real, sim, device)
    model = UNet(width=args.width, cond="norm", n_modalities=2,
                 extra_in=n_extra(kind)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    for step in range(1, args.steps + 1):
        x, y, _, tag, _ = ds.sample(args.batch)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            o = oracle_channels(y, kind)
            xin = x if o is None else torch.cat([x, o], dim=1)
            loss = loss_fn(model(xin, tag).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                ema[k].mul_(0.999).add_(v.float(), alpha=0.001)
        if step % 1000 == 0:
            print(f"  [{kind}] step {step:>5} loss {float(loss):.5f}", flush=True)

    model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in ema.items()})
    out = {}
    for split in args.splits.split(","):
        dl = DataLoader(build("real", split, stride=40), batch_size=64)
        p, t = evaluate(model, dl, device, kind)
        sc = subscores(p, t)
        c = S.measured_channels(t)
        sc["rel_l2_err"] = float(np.mean(S.rel_l2_per_sample(p, t, c)))
        out[split] = sc
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="none,mean,k2,k4,k8")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", default="val_aoa,val_re")
    ap.add_argument("--with-test-aoa", action="store_true",
                    help="match the px2 proxy's training set instead of the 58")
    ap.add_argument("--out", default="runs/info_limit.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    kinds = args.oracle.split(",")
    res = {}
    for kind in kinds:
        print(f"=== oracle: {kind} ({n_extra(kind)} extra channels) ===", flush=True)
        res[kind] = run(kind, args, device)
        json.dump(res, open(args.out, "w"), indent=1)

    # An honest model can only close the gap to the cheated one, and only by getting
    # the cheated component exactly right -- so this is an upper bound, not a target.
    print(f"\n{'oracle':>7} " + " ".join(f"{s:>22}" for s in args.splits.split(",")))
    print(f"{'':>7} " + " ".join(f"{'err':>8}{'score':>7}{'head':>7}"
                                 for _ in args.splits.split(",")))
    base = {s: res[kinds[0]][s]["rel_l2_err"] for s in args.splits.split(",")}
    for kind in kinds:
        row = f"{kind:>7} "
        for s in args.splits.split(","):
            e = res[kind][s]["rel_l2_err"]
            gain = 100.0 * (base[s] - e) / base[s]
            row += f" {e:>8.4f}{res[kind][s]['rel_l2_score']:>7.2f}{gain:>6.1f}%"
        print(row)
    print("\n'head' is the relative rel-L2 reduction against the un-cheated model: "
          "what this\nrecipe gains when that component is HANDED to it. Read it as "
          "the value of the\ncomponent, not as a bound on any forecaster -- the "
          "oracle sees the target, so it\ndoes not measure what is inferable from "
          "the inputs. At 0.0369 of final score per\n10% relative, multiply by "
          "0.0037 to price it.")


if __name__ == "__main__":
    main()
