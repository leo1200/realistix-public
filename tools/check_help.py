#!/usr/bin/env python3
"""Prove every `--help` in tools/ actually renders.

argparse formats each help string with `help % params`, so a bare `%` in prose is a
TypeError raised only when someone asks for help. `train_mixed.py --help` and
`kfold.py --help` had both been broken this way for as long as the strings existed
-- the first by "58% of steps", the second by "~29% of the trajectories" -- and
nothing noticed, because the flags work and only the documentation crashes. That is
the worst place for it: `train_mixed.py` is the trainer the Decision Phase
reproducer calls, and `--help` is how someone who has never seen this repo finds out
what it takes.

Parsed rather than executed, so it costs no imports and catches the file even when
running it would need a GPU.

    python tools/check_help.py
"""

from __future__ import annotations

import ast
import glob
import os
import sys

#: What argparse substitutes. A help string may reference these and nothing else.
PARAMS = dict(prog="x", default="x", type="x", choices="x", dest="x", metavar="x",
              const="x", nargs="x")


def offenders(path: str) -> list[tuple[int, str, str]]:
    try:
        tree = ast.parse(open(path).read())
    except SyntaxError as e:
        return [(e.lineno or 0, "file does not parse", str(e))]
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "add_argument"):
            continue
        for kw in node.keywords:
            if kw.arg != "help":
                continue
            try:
                val = ast.literal_eval(kw.value)
            except Exception:
                continue                      # computed at runtime; not our business
            if not isinstance(val, str):
                continue
            try:
                val % PARAMS
            except Exception as e:
                out.append((node.lineno, str(e), val[:90]))
    return out


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    paths = sorted(glob.glob(os.path.join(here, "*.py")))
    bad = 0
    for p in paths:
        for lineno, why, text in offenders(p):
            bad += 1
            print(f"{os.path.relpath(p, os.path.dirname(here))}:{lineno}: {why}")
            print(f"    {text!r}")
            print("    (escape a literal percent as %% -- argparse formats this string)")
    print(f"\n{len(paths)} file(s) checked -- " +
          ("every --help renders." if not bad else f"{bad} unformattable help string(s)."))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
