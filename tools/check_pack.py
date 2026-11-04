#!/usr/bin/env python3
"""Does the packed submission compute what the run directories compute?

Every measurement in this project scores *run directories*: kfold.py, the in-grid
rig, train_combiner.py all load `best.pt` and evaluate the module. The leaderboard
scores a *packed checkpoint* -- modality affines folded into their GroupNorms,
weights cast to fp16, modules rebuilt from recorded config by submission.py. That
transformation has never been checked against the thing it is supposed to
reproduce; `fold_modality` is asserted exact to 4.8e-7 in a comment, and nothing
tests it end to end.

It matters now because submission 894861 lost 0.159 on the board while the same
recipe measured +0.163 per member in-grid and +0.146 on nine regime folds. If the
packed artefact and the runs disagree, every local number is about a different
model than the board scored.

    python tools/check_pack.py --sub submissions/tke1em8ff
    python tools/check_pack.py --sub submissions/distilled --runs dst_m0,dst_m1,dst_m2

**What it does and does not establish.** It compares the packed
`predict(x, {})` -- prediction AND, since 2026-09-11, `lower`/`upper` -- against the
same ensemble and interval head rebuilt from their checkpoints in fp32 (the head
from the path the pack records, hash-checked). That covers the point forecast and
the intervals -- folding, fp16 casting, module reconstruction, the spread channels
the head reads -- and nothing else: not SPS itself, not runtime, not repeated-call
behaviour. Before the bounds were compared a pack whose head tensors had been
zeroed read MATCH (Codex, 2026-09-11 round 2). A MATCH is "the outputs agree with
the measured modules", not "the pack is correct".

**Result on the champion (2026-09-01, 16 windows of val_aoa).** Against its true
members, mean relative field error **1.98e-4**, max abs 9.2e-5: fp16 folding is
lossless at the scale of the scores, as documented. Against the run directories
named in `model.pth["members"]` it reads **1.01e-2** -- because those bare basenames
resolve to whatever `best.pt` those directories hold today (step 8000), while the
pack was built from step-6000 snapshots. That is a provenance defect, not a packing
defect, and it is why `build_ckpt.py` now records `member_paths` and
`member_sha256`. Packs built before that record the basenames only, and this script
says so in its output rather than silently comparing against the wrong weights.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
RUNS = os.environ.get("REALISTIX_RUNS", "runs")

from realistix.models import build_model  # noqa: E402
from realistix.models.combiner import Combiner  # noqa: E402
from realistix.models.unet import UNet  # noqa: E402


def load_submission(path):
    spec = importlib.util.spec_from_file_location("subm", path)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--runs", default=None,
                    help="comma-separated run dir names; defaults to the member "
                         "checkpoints recorded in the pack's model.pth")
    ap.add_argument("--combiner", default=None, help="run dir holding the combiner")
    ap.add_argument("--ckpt-name", default="best.pt",
                    help="snapshot file to read inside each run dir; ignored when the "
                         "pack records full member paths")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="mean relative field error accepted as a match")
    ap.add_argument("--eval", default=os.environ.get(
                        "CHECK_PACK_EVAL", "eval/val_aoa"),
                    help="Codabench-layout directory holding samples.npz (build one from the "
                         "release with tools/build_eval_npz.py); CHECK_PACK_EVAL sets the default")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    npz = os.path.join(a.eval, "samples.npz")
    if not os.path.exists(npz):
        raise SystemExit(f"{npz} not found: pass --eval / CHECK_PACK_EVAL pointing at a directory "
                         f"made by tools/build_eval_npz.py (this check needs no historical cache)")
    print(f"inputs from {npz}")
    src = np.load(npz)["input"]
    x = src[:a.n].astype(np.float32)

    m = load_submission(os.path.join(a.sub, "submission.py"))
    # The official ingestion program always passes a metadata dict, and so does
    # `run_and_score.py`. Calling with one argument here exercised a signature the
    # leaderboard never uses.
    out = m.predict(x, {})
    packed = out["prediction"].astype(np.float64)[..., :2]
    packed_lo = out["lower"].astype(np.float64)[..., :2]
    packed_hi = out["upper"].astype(np.float64)[..., :2]

    # Which checkpoints *are* the members? `members` records bare parent-directory
    # basenames, so resolving them through `<run>/best.pt` reads whatever that
    # directory holds today -- which is how this check reported a 1.04% mismatch on
    # tke1em8ff, whose members are step6000 snapshots while their run dirs now have
    # step8000 in best.pt. Packs built after 2026-09-01 record the resolved member
    # paths and their hashes; prefer those and fall back to the old guess.
    meta = torch.load(os.path.join(a.sub, "model.pth"), map_location="cpu")
    if a.runs:
        member_paths = [os.path.join(RUNS, n, a.ckpt_name) for n in a.runs.split(",")]
        provenance = f"--runs (via {a.ckpt_name})"
    elif meta.get("member_paths"):
        member_paths = list(meta["member_paths"])
        provenance = "member_paths recorded in model.pth"
    else:
        member_paths = [os.path.join(RUNS, n, a.ckpt_name)
                        for n in meta.get("members", [])]
        provenance = (f"members basenames in model.pth (via {a.ckpt_name}) -- this pack "
                      f"predates path provenance, so the snapshot identity is a guess")
    if not member_paths:
        raise SystemExit("no member checkpoints to compare against; pass --runs")

    digests = meta.get("member_sha256")
    if digests:
        import hashlib
        for p, want in zip(member_paths, digests):
            got = hashlib.sha256(open(p, "rb").read()).hexdigest()
            if got != want:
                raise SystemExit(
                    f"member {p} hashes {got[:12]}, pack recorded {want[:12]} -- the "
                    f"checkpoint changed since the pack was built")

    # The same computation from the run directories, unfolded and in fp32.
    xt = torch.from_numpy(x).to(device)
    uv = xt[..., :2]
    scale = m._window_scale(uv)
    s = scale[:, None, None, None, None]
    xin = (uv / s).permute(0, 1, 4, 2, 3).reshape(-1, 40, 32, 64).contiguous()
    outs = []
    for path in member_paths:
        ck = torch.load(path, map_location=device)
        mod = build_model(ck["args"])
        mod.load_state_dict(ck["model"])
        mod = mod.to(device).eval()
        with torch.no_grad():
            outs.append(mod(xin, torch.zeros(xin.shape[0], dtype=torch.long,
                                             device=device)).float())
    stack = torch.stack(outs)
    y = stack.mean(0)
    # Reconstruct the combiner the pack itself records. This used to fire only when
    # the caller passed --combiner by hand, so a perfectly good combined pack read
    # MISMATCH (measured on submissions/distilled: 4.25e-3 without it, 2.08e-4 with)
    # -- and now that the checker exits non-zero, that would stop a correct build.
    if a.combiner:
        cck = torch.load(os.path.join(RUNS, a.combiner, "best.pt"), map_location=device)
        ca = cck["args"]
        comb = Combiner(n_members=ca["n_models"], width=ca["width"], depth=ca["depth"],
                        spread=not ca["no_spread"], window=not ca["no_window"])
        comb.load_state_dict(cck["combiner"])
        comb = comb.to(device).eval()
        with torch.no_grad():
            y = comb(stack, xin.float())
    elif meta.get("combiner") is not None:
        cfg = meta.get("combiner_cfg") or {}
        comb = Combiner(n_members=cfg.get("n_members", len(member_paths)),
                        width=cfg["width"], depth=cfg["depth"],
                        spread=cfg.get("spread", True), window=cfg.get("window", True))
        comb.load_state_dict({k: v.float() for k, v in meta["combiner"].items()})
        comb = comb.to(device).eval()
        with torch.no_grad():
            y = comb(stack, xin.float())
        print(f"  combiner reconstructed from the pack "
              f"({meta.get('combiner_path', 'no path recorded')})")
    # An inference-time transform of the ensemble mean (submissions/k4g's member-variance-matched
    # fluctuation gain, module constant GAIN_DOSE) is part of the shipped function, so the fp32
    # reference has to apply it too -- otherwise the parity check reports the transform as a defect.
    # The head still reads the PLAIN mean, exactly as the submission does.
    gain_dose = float(getattr(m, "GAIN_DOSE", 0.0) or 0.0)
    y_plain = y
    if gain_dose > 0:
        y = m._matched_gain(stack, y)
        print(f"  inference transform: GAIN_DOSE={gain_dose} applied to the reference as well")
    # A second inference-time transform (fm3all, 2026-09-22): the component-wise blend with a
    # distilled flow donor. Same principle as the gain above -- it is part of the shipped
    # function, so the reference must apply it or parity would report the transform as a defect.
    # The donor is rebuilt HERE from the training checkpoint the pack records, not read out of
    # the pack, so a corrupted or mis-folded packed donor still fails. Inert for every pack
    # without a donor, which is every pack built before today.
    flow_w = float(meta.get("flow_w") or 0.0)
    if meta.get("flow_student") is not None and flow_w > 0:
        import hashlib as _hl
        src = meta.get("flow_student_src")
        if not src or not os.path.exists(src):
            raise SystemExit(f"pack carries a flow donor but its source checkpoint is missing: {src}")
        if _hl.sha256(open(src, "rb").read()).hexdigest() != meta.get("flow_student_sha256"):
            raise SystemExit(f"flow donor source {src} does not hash to the digest the pack recorded")
        dck = torch.load(src, map_location="cpu")
        dsd = {k[len("trunk."):]: v for k, v in dck["model"].items()}
        donor = UNet(width=meta["width"],
                     mults=tuple(int(v) for v in str(meta.get("mults") or "1,2,4").split(",")),
                     blocks_per_level=int(meta.get("blocks") or 2),
                     n_modalities=2, cond="norm", extra_in=1, predict_delta=True, anchor="last")
        donor.load_state_dict(dsd)
        donor = donor.to(device).float().eval()
        # extra_in channels are appended to the window by the CALLER (research
        # model_student.Student does exactly this), not passed as log_scale=.
        ls = torch.log(scale).float()[:, None, None, None].expand(-1, 1, *xin.shape[-2:])
        with torch.no_grad():
            f = donor(torch.cat([xin.float(), ls], 1),
                      torch.zeros(xin.shape[0], dtype=torch.long, device=device))
        y = m._flow_blend(y, f.float(), flow_w)
        print(f"  inference transform: flow donor blend w={flow_w} applied to the reference as well "
              f"(donor rebuilt from {os.path.basename(src)})")
    ref = (y.reshape(-1, 20, 2, 32, 64).permute(0, 1, 3, 4, 2) * s
           ).cpu().numpy().astype(np.float64)

    # The intervals, from the head checkpoint the pack records, in fp32, with the
    # spread channels computed the way train_final_head fits them.
    ref_lo = ref_hi = None
    hpath = meta.get("head_path")
    if hpath and os.path.exists(hpath) and meta.get("combiner") is None:
        import hashlib
        if meta.get("head_sha256") and hashlib.sha256(open(hpath, "rb").read()).hexdigest() != meta["head_sha256"]:
            raise SystemExit(f"head {hpath} does not hash to the digest the pack recorded")
        from realistix.models.uncertainty import UncertaintyHead
        from realistix.losses import SIGMA_GLOBAL
        hck = torch.load(hpath, map_location=device)
        ha = hck["args"]
        head = UncertaintyHead(width=ha["width"], extra_channels=80 if ha.get("spread") else 0,
                               offset=bool(ha.get("offset", False))).to(device).eval()
        head.load_state_dict(hck["head"])
        with torch.no_grad():
            extra = None
            if head.extra_channels:
                r = stack.std(0, unbiased=True) * scale[:, None, None, None] / SIGMA_GLOBAL
                extra = torch.cat([r, torch.log(r.clamp_min(1e-4))], dim=1)
            below, above = head(xin.float(), y_plain, torch.log(scale), extra)
        yy = y_plain.reshape(-1, 20, 2, 32, 64).permute(0, 1, 3, 4, 2) * s
        bb = below.float().reshape(-1, 20, 2, 32, 64).permute(0, 1, 3, 4, 2)
        aa = above.float().reshape(-1, 20, 2, 32, 64).permute(0, 1, 3, 4, 2)
        # The bounds bracket the PLAIN mean, but the shipped containment guard
        # (`torch.minimum(lo, pred)`, `torch.maximum(hi, pred)`) is applied against the
        # SHIPPED point forecast -- the gained one when GAIN_DOSE > 0. Until 2026-09-17 this
        # reference guarded against the plain mean, so a gained pack's interval discrepancy
        # was partly the guard and not only fp16 (Codex audit, 2026-09-16).
        yg = y.reshape(-1, 20, 2, 32, 64).permute(0, 1, 3, 4, 2) * s
        ref_lo = torch.minimum(yy - bb, yg).cpu().numpy().astype(np.float64)
        ref_hi = torch.maximum(yy + aa, yg).cpu().numpy().astype(np.float64)
    elif meta.get("combiner") is None:
        print(f"  head checkpoint not available ({hpath}); intervals NOT compared")

    d = np.abs(packed - ref)
    denom = np.abs(ref).mean()
    rel = d.mean() / denom
    print(f"{a.n} windows, {a.sub}")
    print(f"  members from {provenance}")
    for p in member_paths:
        print(f"    {p}")
    print(f"  packed vs runs: max abs {d.max():.3e}  mean abs {d.mean():.3e}")
    print(f"  relative to field magnitude {denom:.3e}: mean {rel:.3e}")
    ok = rel < a.tol
    if ref_lo is not None:
        # Widths are what SPS pays for; compare the half-widths relative to the mean
        # width so a head that emits nothing (zeroed) or something else fails here.
        w_ref = (ref_hi - ref_lo); w_pk = (packed_hi - packed_lo)
        dw = np.abs(w_pk - w_ref); rel_w = dw.mean() / max(w_ref.mean(), 1e-12)
        d_lo = np.abs(packed_lo - ref_lo).mean() / denom; d_hi = np.abs(packed_hi - ref_hi).mean() / denom
        print(f"  intervals vs head checkpoint {os.path.basename(os.path.dirname(hpath))}: "
              f"width mean {w_ref.mean():.3e}, relative width error {rel_w:.3e}, "
              f"lower {d_lo:.3e} / upper {d_hi:.3e} relative to field magnitude")
        ok = ok and rel_w < a.tol * 10 and d_lo < a.tol and d_hi < a.tol
    if ok:
        print("  MATCH" + ("" if ref_lo is not None else " (prediction only)"))
        return
    # Exit non-zero. This printed MISMATCH and returned success until 2026-09-01, so
    # `build_candidate.sh`'s `|| true` and every by-eye reading of the log agreed the
    # pack was verified when the checker had just said the opposite.
    print("  MISMATCH -- the packed model is not the measured model")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
