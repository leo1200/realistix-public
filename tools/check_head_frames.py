#!/usr/bin/env python3
"""Provenance checks for interval-head / combiner inputs.

Two questions, both of which used to be answered by looking at file timestamps:

  * did this head honour the tail holdout when it sampled its training windows?
  * did the members underneath it train under the same holdout as this run?

Timestamps answered neither. `touch` defeated the first, and any unrelated edit to
`train_final_head.py` made it reject correctly-built heads. Both checks now read what
the checkpoint recorded about itself.

    python tools/check_head_frames.py <head.pt>      # exit 0 if tail-clean
"""

from __future__ import annotations

import os
import sys

import torch

#: Holdout axes a run records, so a consumer can prove it saw the same data.
HOLDOUT_KEYS = ("REALISTIX_HELDOUT_AOA", "REALISTIX_HELDOUT_RE",
                "REALISTIX_HELDOUT_CELLS", "REALISTIX_HELDOUT_TAIL",
                "REALISTIX_SIM_KEEPS_CELLS")


def head_is_tail_clean(path):
    """(ok, message) for the head checkpoint at ``path``."""
    try:
        a = torch.load(path, map_location="cpu")["args"]
    except Exception as e:                          # noqa: BLE001 - reported, not raised
        return False, f"cannot read {path}: {e}"
    hf = a.get("head_frames")
    if hf is None:
        return False, (f"{path} predates the head_frames stamp (2026-09-01), so it may "
                       f"have trained on the evaluation tail. Delete it to remeasure, "
                       f"or use a different name.")
    if hf != "head":
        return False, (f"{path} was trained with frames={hf!r}, not 'head': it sampled "
                       f"the evaluation tail.")
    return True, f"{path}: tail-clean (head_frames='head')"


def members_match_holdout(paths, strict=False):
    """Do these member checkpoints record the holdout this process is running under?

    A postprocessor (interval head, combiner) that samples correctly is still not
    held out if the members it sits on top of trained on the evaluation data. Both
    scripts accepted arbitrary checkpoints until 2026-09-01, so an all-data member
    could be fed to a purported held-out experiment and nothing would say so.

    Returns ``(errors, warnings)``. Errors are genuine disagreements and should stop
    a run. Warnings are checkpoints that predate a given record and therefore cannot
    be checked -- which is every member trained before 2026-09-01, so making those
    fatal would break every existing ensemble for no new information. ``strict=True``
    promotes them to errors, for a study that must prove its provenance.
    """
    bad, warn = [], []
    for p in paths:
        try:
            a = torch.load(p, map_location="cpu")["args"]
        except Exception as e:                      # noqa: BLE001 - reported, not raised
            bad.append(f"{p}: cannot read args ({e})")
            continue
        unknown = []
        for k in HOLDOUT_KEYS:
            want = os.environ.get(k) or ""
            got = a.get(k)
            if got is None:
                unknown.append(k)
            elif (got or "") != want:
                bad.append(f"{p}: trained with {k}={got!r}, this run has {want!r}")
        if unknown:
            (bad if strict else warn).append(
                f"{p}: predates the {', '.join(unknown)} record, so its holdout "
                f"cannot be verified here")
    return bad, warn


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {os.path.basename(__file__)} <head.pt>")
    ok, msg = head_is_tail_clean(sys.argv[1])
    print(msg)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
