#!/usr/bin/env python3
"""Assemble the single ``model.pth`` a submission loads, from training run dirs.

The submission carries one file: the ensemble members' weights plus the interval
head, in fp16. Everything the submission needs to rebuild the modules -- widths,
whether FiLM conditioning is present, whether the head reads ensemble spread --
travels in the same dict, so ``submission.py`` never has to be edited in step with
a config change.

fp16 is a size decision, not an accuracy one: three width-96 members are 278 MB in
fp32 against a 256 MB extracted cap. It was verified lossless in the scored
metrics (subscores identical to four decimals) before it shipped.

    python tools/build_ckpt.py --head runs/hd_spread3/best.pt \
        --out submissions/ensemble96_spread/model.pth
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def fold_modality(sd, row=0):
    """Fold the per-modality affine into the GroupNorm it follows.

    Inference always selects one modality, and a fixed affine applied after a
    normalisation is that normalisation with different parameters:

        GN(x; w, c) * (1 + g) + b  ==  GN(x; w * (1 + g), c * (1 + g) + b)

    So domain-specific normalisation costs nothing at inference -- which matters,
    because shipping it unfolded cost 1.5 ms/sample and 0.66 of time_score, and
    time_score's weight is ~0.09 rather than the ~0.00 two earlier fits suggested.
    The shipped network becomes a plain U-Net; the conditioning stays a
    training-time device, which is all it ever needed to be.
    """
    out, folded = {}, 0
    for k, v in sd.items():
        if k.endswith((".f1.weight", ".f2.weight")):
            continue                                    # consumed below
        out[k] = v
    for k in list(sd):
        if not k.endswith((".f1.weight", ".f2.weight")):
            continue
        norm = k.replace(".f1.weight", ".n1").replace(".f2.weight", ".n2")
        g, b = sd[k][row].float().chunk(2)
        out[norm + ".weight"] = (sd[norm + ".weight"].float() * (1 + g)).to(sd[norm + ".weight"].dtype)
        out[norm + ".bias"] = (sd[norm + ".bias"].float() * (1 + g) + b).to(sd[norm + ".bias"].dtype)
        folded += 1
    return out, folded


def half(sd):
    return {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}


def fold_decomposed_output(sd, args):
    """Merge mean + temporally centered fluctuation convolutions before fp16."""
    if not args.get("decompose"):
        return sd
    if args.get("anchor", "last") != "last":
        raise SystemExit("decomposed output folding currently requires anchor=last")
    if any(args.get(k) for k in ("out_gain", "out_cumsum", "warp", "mimo")):
        raise SystemExit("decomposed output folding requires the ordinary output path")
    result = dict(sd)
    for suffix in ("weight", "bias"):
        raw = result["out." + suffix].float()
        mean = result.pop("out_mean." + suffix).float()
        if raw.shape[0] != 20 * mean.shape[0] or raw.shape[1:] != mean.shape[1:]:
            raise SystemExit("decomposed output folding requires the 20-frame interface")
        fluct = raw.reshape(20, *mean.shape)
        folded = fluct - fluct.mean(0, keepdim=True) + mean.unsqueeze(0)
        result["out." + suffix] = folded.reshape_as(raw).to(sd["out." + suffix].dtype)
    return result


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_cooperative_companions(path, args, members):
    """A cooperative member is tied to the actual fixed ensemble used in training."""
    if not args.get("cooperate_with"):
        return
    companions = sorted(os.path.realpath(q) for q in args["cooperate_with"].split(","))
    packed = sorted(os.path.realpath(q) for q in members
                    if os.path.realpath(q) != os.path.realpath(path))
    if companions != packed:
        raise SystemExit(f"{path} was trained to cooperate with {companions}, "
                         f"but this pack contains {packed}")
    for q, digest in args.get("cooperate_sha256", {}).items():
        if _sha256(q) != digest:
            raise SystemExit(f"cooperative companion {q} changed since {path} was trained")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, help="best.pt from train_final_head")
    ap.add_argument("--runs", default=None,
                    help="ensemble glob; defaults to the paths recorded in the head")
    ap.add_argument("--n-models", type=int, default=3)
    ap.add_argument("--allow-member-mismatch", action="store_true",
                    help="pack --runs members even though the head was calibrated "
                         "against different ones; the intervals will not match")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hck = torch.load(args.head, map_location="cpu")
    ha = hck["args"]
    if ha.get('oof_members'):
        expected = {os.path.realpath(q) for q in ha['oof_members'].split(',')}
        recorded = ha.get('oof_sha256', {})
        if expected != set(recorded):
            raise SystemExit('OOF head must record every calibration member hash')
        for q, digest in recorded.items():
            if not os.path.isfile(q) or _sha256(q) != digest:
                raise SystemExit(f'OOF calibration member missing or changed: {q}')
    paths = (sorted(glob.glob(os.path.join(args.runs, "best.pt")))[:args.n_models]
             if args.runs else hck["paths"])

    # The head was calibrated against a specific ensemble. Packing it around a
    # different one produces an artefact whose point predictions still match those
    # members -- so check_pack says MATCH -- while its intervals are calibrated for
    # weights that are not in the pack. --runs silently allowed exactly that.
    if args.runs:
        want = [os.path.realpath(p) for p in hck["paths"]]
        got = [os.path.realpath(p) for p in paths]
        if want != got and not args.allow_member_mismatch:
            raise SystemExit(
                "--runs resolves to different members than the head was trained on:\n"
                + "\n".join(f"  head: {w}\n  runs: {g}" for w, g in zip(want, got))
                + "\nThe interval head would be calibrated for weights that are not "
                  "in this pack. Pass --allow-member-mismatch if that is intended.")

    # The head can read spread from members that are not themselves ensembled
    # (`--spread-runs`). Those weights are not packed and submission.py cannot
    # reconstruct them, so the shipped head would see a different spread input than
    # the one it was fitted on.
    if hck.get("extra_paths"):
        raise SystemExit(
            f"the head was trained with {len(hck['extra_paths'])} spread-only member(s) "
            f"({', '.join(hck['extra_paths'])}); submission.py cannot reconstruct them, "
            f"so its spread input would differ from the one the head was fitted on")

    # Content, not path: a head fitted on 2026-09-11 or later records its members'
    # digests at fit time. A member overwritten since then keeps its path and would
    # pass the path comparison above with intervals calibrated for other weights.
    if hck.get("member_sha256") is not None and not args.allow_member_mismatch:
        now = [_sha256(os.path.realpath(p)) for p in paths]
        if now != list(hck["member_sha256"]):
            raise SystemExit("the members on disk are not the ones the head was fitted on "
                             "(SHA256 differs); refit the head or pass --allow-member-mismatch")
        print(f"head fit-time member hashes match the {len(paths)} packed members")

    # Options the training scripts accept but `submission.py` does not reconstruct.
    # `anchor=wmean` is the dangerous one: packed inference treats every anchor that
    # is not `blend` as `last`, so the pack loads cleanly and computes a different
    # function (measured 2026-09-01 on wm_a05_re2_s0: mean abs 0.774, max 4.107
    # normalized units, matching the wmean->last swap to 4.8e-7). The rest fail
    # strict loading, which is merely confusing rather than silent.
    SHIPPABLE_ANCHORS = ("last", "blend")

    models, folded_any = [], False
    width = film = cond = mults = blocks = None
    for p in paths:
        ck = torch.load(p, map_location="cpu")
        a = ck["args"]
        if a.get('output_adapter'):
            from realistix.output_adapter import verify_constraint
            base_path=a.get('init_from')
            if not base_path or _sha256(base_path)!=a.get('output_adapter_base_sha256'):
                raise SystemExit('output adapter base is missing or changed')
            base=torch.load(str(base_path),map_location='cpu',mmap=True)
            verify_constraint(base['model'],ck['model'],a.get('output_adapter_mode','fluctuation'))
        check_cooperative_companions(p, a, paths)
        if a.get("norm_groups", 8) != 8:
            raise SystemExit(
                f"{p} uses norm_groups={a['norm_groups']}; existing submission.py "
                "reconstructs 8 groups. Add inference support and verify pack parity "
                "before packaging this architecture (state_dict shapes alone cannot detect it).")
        if a.get("anchor", "last") not in SHIPPABLE_ANCHORS:
            raise SystemExit(
                f"{p} trained with anchor={a['anchor']!r}, which submission.py does "
                f"not reconstruct -- it would silently ship as 'last'. Shippable "
                f"anchors: {', '.join(SHIPPABLE_ANCHORS)}")
        # mults and blocks join the key because the submission has to rebuild the
        # module before loading weights, and it cannot infer either from a state
        # dict. They were absent until 2026-08-17, when the op-site reduction work
        # made non-default values shippable for the first time; a mismatch used to
        # surface only as a load_state_dict failure at build time.
        # `anchor` belongs in the identity key, not just the payload. Members are
        # grouped by this tuple, so leaving it out would let a `blend` member and a
        # `last` member pack into one ensemble that loads under a single
        # architecture -- silently, and with 20 parameters landing on the wrong
        # network.
        key = (a["width"], bool(a.get("film")), a.get("cond", "enc"),
               str(a.get("mults", "1,2,4")), int(a.get("blocks", 2)),
               a.get("anchor", "last"))
        if width is None:
            width, film, cond, mults, blocks, anchor = key
        elif key != (width, film, cond, mults, blocks, anchor):
            raise SystemExit(f"{p} disagrees on architecture with the first member:\n"
                             f"  first={(width, film, cond, mults, blocks, anchor)}\n  this ={key}")
        sd, n_folded = fold_modality(fold_decomposed_output(ck["model"], a))
        if n_folded:
            folded_any = True
        models.append(half(sd))

    out = {
        "models": models,
        "width": width,
        "mults": mults,
        "blocks": blocks,
        "anchor": anchor,
        # After folding there is no modality path left in the shipped network.
        "film": film and not folded_any,
        "cond": "none" if folded_any else cond,
        "head": half(hck["head"]),
        "head_width": ha["width"],
        "head_spread": bool(ha.get("spread")),
        # Resolved, because the ensemble is usually assembled as a directory of
        # symlinks and "m1, m2, m3" is not a provenance record.
        "members": [os.path.basename(os.path.dirname(os.path.realpath(p))) for p in paths],
        # ...and the basename is not one either, which cost a day of audit time: a
        # member symlinked to `step6000.pt` records the same directory name as that
        # directory's current `best.pt` (step 8000), so `check_pack.py` compared the
        # pack against checkpoints it was not built from and reported 1.04% drift as
        # if fp16 folding were lossy. The exact file and its hash pin the snapshot.
        "member_paths": [os.path.realpath(p) for p in paths],
        "member_sha256": [_sha256(p) for p in paths],
        # Provenance, so the packer can tell "built from the current head" from
        # "left over because build_ckpt failed" without guessing from timestamps.
        "head_path": os.path.realpath(args.head),
        "head_mtime": os.path.getmtime(args.head),
        # mtime says when a file was written, not which file it is. Overwriting a
        # checkpoint in place -- an ordinary retraining move -- leaves the path and
        # the recorded time plausible while the weights are different.
        "head_sha256": _sha256(args.head),
        "head_oof_sha256": ha.get('oof_sha256', {}),
        # What the head was CALIBRATED against, which is not necessarily what got
        # packed: --runs can override it. Recording both lets a later check tell a
        # legitimate repack from a head paired with the wrong ensemble.
        "head_member_paths": [os.path.realpath(q) for q in hck["paths"]],
        "head_member_sha256": [_sha256(q) for q in hck["paths"] if os.path.exists(q)],
    }

    # The learned combiner, if the head was trained on top of one. Its config
    # travels with its weights for the same reason mults and blocks do: the
    # submission rebuilds the module before loading, and cannot infer the shape
    # from a state dict. Taken from the *head's* record rather than from a flag,
    # so a head fitted around the combined prediction can never be packed with a
    # different combiner than it was fitted to -- or with none at all, which would
    # load cleanly and quietly draw intervals around the wrong prediction.
    cpath = ha.get("combiner")
    if cpath:
        cck = torch.load(cpath, map_location="cpu")
        ca = cck["args"]
        out["combiner"] = half(cck["combiner"])
        out["combiner_cfg"] = dict(n_members=ca["n_models"], width=ca["width"],
                                   depth=ca["depth"], spread=not ca["no_spread"],
                                   window=not ca["no_window"])
        out["combiner_path"] = os.path.realpath(cpath)
        out["combiner_sha256"] = _sha256(cpath)
        if ca["n_models"] != len(models):
            raise SystemExit(f"combiner expects {ca['n_models']} members, "
                             f"packing {len(models)}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    mb = os.path.getsize(args.out) / 1e6
    print(f"{len(models)} x width-{width} (trained film={film} cond={cond}"
          f"{', folded to a plain U-Net' if folded_any else ''}) "
          f"+ head width {ha['width']} "
          f"(spread={out['head_spread']})")
    print(f"members: {', '.join(out['members'])}")
    if "combiner" in out:
        print(f"combiner: {out['combiner_cfg']}")
    print(f"wrote {args.out}  {mb:.1f} MB")


if __name__ == "__main__":
    main()
