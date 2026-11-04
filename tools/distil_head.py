#!/usr/bin/env python3
"""Distil an averaged interval-head ensemble into ONE head of the same architecture.

The shipped interval estimator since 2026-09-17 (validated on the board as 929146):
six SPS heads fitted independently on the frozen forecasters (``tools/train_final_head.py``,
seeds 0/11/21/31/41/51) are *training artefacts*; their per-element ``below``/``above``
bounds are averaged in linear units and a single student of the same architecture is
regressed onto that average in log space. The student ships; the teachers do not.

Why. Averaging two heads' OUTPUTS paid +0.0807 SPS on the board (926634) -- the head's
policy is optimal (FINDINGS §2b) and the gain is estimator-variance reduction -- but a
second head costs ~1 ms of inference. The averaged bound function is a fixed, smooth
target, so one student carries most of it at one head's cost: fold mean +0.1223 SPS vs
the single head against the pair's +0.0999 (research/distil_2026-09-16/PREREG.md), and
board SPS 43.9453 vs the pair's 43.9414 (CANDIDATE_k4a2.md). Teacher-0 initialisation
and the asymmetric loss (under-width penalised 3x: a too-narrow interval loses coverage
outright, a too-wide one only pays exp(-delta/sigma)) are load-bearing at the 4000-step
budget (fresh init captured 40 % of the teacher average; this recipe 82 %).

The training loop is the research script (research/distil_2026-09-16/distil.py) line for
line, so a replay with the same teachers and seed reproduces the shipped student's
tensors bitwise on the same card class -- that replay is the positive control of
``tools/check_recipe.py --replay-student``. Do not reorder the sampler or RNG calls.

    python tools/distil_head.py --teachers a/best.pt,b/best.pt,... --out DIR
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "src"), HERE, os.path.join(HERE, "..", "starting_kit")]
import torch, torch.nn.functional as F  # noqa: E402

from realistix.data import HELDOUT_TAIL  # noqa: E402
from realistix.gpudata import GPUWindows  # noqa: E402
from realistix.splits import partition  # noqa: E402
from realistix.evaluate import amp_dtype_for  # noqa: E402
from realistix.models.uncertainty import UncertaintyHead  # noqa: E402
from train_final_head import Ensemble, spread_channels  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Exposed so tools/check_recipe.py parses train_final.sh's STUDENT literal through
    the same definitions this trainer uses. Defaults are the shipped student's (A2)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", required=True,
                    help="comma-separated best.pt of the teacher heads, all fitted on the same members")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train-split", default="all")
    # SPS never reads an element whose target is exactly zero -- inside the airfoil body
    # and outside the PIV field of view (starting_kit/scoring.py; `scored` in
    # train_final_head.py is exactly this rule). The TEACHERS are trained with that mask,
    # so they receive no gradient there and whatever they emit is arbitrary. The student
    # regresses onto them over EVERY element, so ~10.1% of its loss terms (measured on the
    # release) imitate that arbitrary output. Off by default: the shipped student is the
    # unmasked one and must stay bit-reproducible.
    ap.add_argument("--mask-unscored", action="store_true",
                    help="drop exactly-zero-target elements from the distillation loss")
    ap.add_argument("--init-teacher", type=int, default=0,
                    help="start from this teacher index instead of a fresh init (-1)")
    ap.add_argument("--under-weight", type=float, default=3.0,
                    help="extra penalty on being NARROWER than the teacher average")
    return ap


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(os.path.realpath(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    a = build_parser().parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(a.seed)

    tpaths = a.teachers.split(",")
    if len(set(os.path.realpath(p) for p in tpaths)) != len(tpaths):
        raise SystemExit("the same teacher is listed twice; duplicated teachers are not an ensemble")
    cks = [torch.load(p, map_location=dev) for p in tpaths]
    # Every teacher must sit on the same frozen forecasters, or their bounds are not
    # estimates of one quantity and averaging them means nothing.
    assert all(c["member_sha256"] == cks[0]["member_sha256"] for c in cks), "teacher members differ"
    if len({c["args"].get("seed") for c in cks}) != len(cks):
        raise SystemExit("two teachers record the same head seed")
    teachers = []
    for c in cks:
        h = UncertaintyHead(width=a.width, extra_channels=80).to(dev).eval()
        h.load_state_dict(c["head"])
        for q in h.parameters():
            q.requires_grad_(False)
        teachers.append(h)
    ens = Ensemble(cks[0]["paths"], dev)

    part = partition("real")
    trajs = (part["train"] + part["val_re"] + part["val_aoa"] + part["test_aoa"]
             if a.train_split == "all" else part[a.train_split])
    frames = "head" if HELDOUT_TAIL else "all"
    train_ds = GPUWindows(trajs, "real", dev, frames=frames)
    print(f"{len(tpaths)} teachers, {len(trajs)} training trajectories, frames={frames}", flush=True)

    student = UncertaintyHead(width=a.width, extra_channels=80).to(dev)
    if a.init_teacher >= 0:
        # Starting from a teacher means the student only has to move toward the
        # average, not rediscover the whole function from noise.
        student.load_state_dict(cks[a.init_teacher]["head"])
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    hist, t0 = [], time.time()
    for step in range(1, a.steps + 1):
        x, y, scale, _, _ = train_ds.sample(a.batch)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype_for(dev)):
            p, sd = ens.mean_std(x)
        p = p.float()
        extra = spread_channels(sd, scale)
        ls = torch.log(scale)
        with torch.no_grad():
            bt, at = zip(*[h(x, p, ls, extra) for h in teachers])
            bt = torch.stack(bt).mean(0)          # the averaged bounds, in linear units --
            at = torch.stack(at).mean(0)          # averaging logs would be a different target
        bs, as_ = student(x, p, ls, extra)
        # Same construction as train_final_head.py:485, so the student is fitted on
        # exactly the elements its teachers were rewarded on.
        scored = (y.abs() * scale[:, None, None, None] > 0).float() if a.mask_unscored else None
        # Log space is the head's own parameterisation (d = exp(log_d)), so this
        # regresses the student's pre-activation directly and weights the two orders of
        # magnitude the widths span evenly instead of only fitting the wake.
        if a.under_weight == 1.0:
            if scored is None:
                loss = F.mse_loss(torch.log(bs), torch.log(bt)) + F.mse_loss(torch.log(as_), torch.log(at))
            else:
                n = scored.sum().clamp_min(1.0)
                loss = (((torch.log(bs) - torch.log(bt)).pow(2)
                         + (torch.log(as_) - torch.log(at)).pow(2)) * scored).sum() / n
        else:
            # The reward is not symmetric in width: too narrow drops coverage outright,
            # too wide only pays exp(-delta/sigma). Penalise the dangerous side harder.
            loss = 0.0
            for s_, t_ in ((bs, bt), (as_, at)):
                d = torch.log(s_) - torch.log(t_)
                w = torch.where(d < 0, a.under_weight, 1.0) * d.pow(2)
                loss = loss + (w.mean() if scored is None
                               else (w * scored).sum() / scored.sum().clamp_min(1.0))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step(); sch.step()
        if step % 500 == 0:
            with torch.no_grad():
                rel = ((bs + as_) / (bt + at)).mean().item()
            hist.append({"step": step, "loss": loss.item(), "width_ratio_vs_teacher": rel})
            print(f"[distil] {step:>5} loss={loss.item():.5f} width/teacher={rel:.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            json.dump(hist, open(os.path.join(a.out, "history.json"), "w"), indent=1)

    # No checkpoint selection: a fixed step, like the teachers' --fixed-step 4000, so
    # nothing is chosen on a validation split. `build_ckpt` reads `head_spread` from the
    # head checkpoint's own args, so the shape facts the packer needs are recorded here,
    # taken from the teachers this student was fitted against.
    sargs = dict(vars(a))
    sargs["spread"] = True
    sargs["width"] = a.width
    sargs["tau_frac"] = cks[0]["args"].get("tau_frac")
    sargs["n_models"] = cks[0]["args"].get("n_models")
    sargs["head_frames"] = cks[0]["args"].get("head_frames")
    torch.save({"head": student.state_dict(), "args": sargs, "paths": cks[0]["paths"],
                "extra_paths": [], "step": a.steps, "teachers": tpaths,
                "teacher_sha256": [sha256(q) for q in tpaths],
                "teacher_seeds": [c["args"].get("seed") for c in cks],
                "member_sha256": cks[0]["member_sha256"]},
               os.path.join(a.out, "best.pt"))
    print("saved", os.path.join(a.out, "best.pt"), flush=True)


if __name__ == "__main__":
    main()
