#!/usr/bin/env python3
"""Apply a fold's holdout to the environment *before* ``realistix.splits`` is imported.

``splits.py`` reads ``REALISTIX_HELDOUT_RE``, ``REALISTIX_HELDOUT_AOA`` and
``REALISTIX_HELDOUT_CELLS`` exactly once, into module-level constants, so a tool that
imports ``realistix.data`` at the top of the file and then assigns those variables
inside ``main()`` is assigning them too late. That was the state of five diagnostic
tools until 2026-09-01, and because the failure is silent every one of their four
"mimic folds" scored the same 175 windows of the default AoA=20 holdout -- including,
for some folds, cells that fold's own members had trained on.

Import this module and call :func:`apply_fold` *above* the first ``from realistix``
import. It depends on nothing but the standard library and ``kfold``, which reads only
``realistix.weights`` and therefore does not pull ``splits`` in.

    import foldenv                                  # noqa: E402
    foldenv.apply_fold_from_argv()                  # before the line below
    from realistix.data import build                 # noqa: E402

``realistix.splits.partition`` now raises if these variables change after import, so a
tool that forgets this will fail loudly instead of quietly answering for AoA=20.
"""

from __future__ import annotations

import os
import sys

from kfold import FOLD_SETS

#: Cleared alongside the cell list, because a mimic fold holds out cells by name and
#: must not additionally inherit the default Reynolds/angle holdout.
_KEYS = ("REALISTIX_HELDOUT_AOA", "REALISTIX_HELDOUT_RE", "REALISTIX_HELDOUT_CELLS")


def fold_settings(name: str) -> dict[str, str]:
    """All holdout axes of a named fold, including angle and Reynolds folds."""
    for folds in FOLD_SETS.values():
        for f in folds:
            if f["name"] == name:
                return {key: f["env"].get(key, "") for key in _KEYS}
    known = sorted(f["name"] for folds in FOLD_SETS.values() for f in folds)
    raise SystemExit(f"unknown fold {name!r}; known folds: {', '.join(known)}")


def fold_cells(name: str) -> str:
    """The cell list, empty for a fold defined by whole angles/Reynolds columns."""
    return fold_settings(name)["REALISTIX_HELDOUT_CELLS"]


def apply_fold(name: str | None) -> str:
    """Put ``name``'s holdout in the environment. Returns the cell list applied."""
    if not name:
        return ""
    if "realistix.splits" in sys.modules:
        raise RuntimeError(
            "realistix.splits is already imported, so applying a fold here would be "
            "silently ignored. Move this call above the first realistix import.")
    settings = fold_settings(name)
    os.environ.update(settings)
    return settings["REALISTIX_HELDOUT_CELLS"]


def argv_folds(*flags: str) -> list[str]:
    """Every fold name given under any of ``flags``, in order, straight from argv.

    Handles both ``--fold NAME`` and an ``nargs="*"`` style ``--folds A B C``; the
    latter is why this cannot just take the single token after the flag.
    """
    names: list[str] = []
    for flag in flags or ("--fold",):
        for i, a in enumerate(sys.argv):
            if a.startswith(flag + "="):
                names.append(a.split("=", 1)[1])
            elif a == flag:
                for v in sys.argv[i + 1:]:
                    if v.startswith("-"):
                        break
                    names.append(v)
    return names


def apply_fold_from_argv(*flags: str) -> str:
    """Read the fold out of ``sys.argv`` and apply it, before any realistix import.

    Deliberately not argparse: the value is needed above the imports that argparse
    would sit below, and the tool parses its own arguments properly afterwards. When
    several folds are named only the **first** is applied — a single process can hold
    exactly one holdout, and a tool that wants more must spawn with
    :func:`rerun_per_fold`.
    """
    names = argv_folds(*flags)
    return apply_fold(names[0]) if names else ""


def rerun_per_fold(folds, base_argv=None, flag: str = "--fold", extra=()) -> None:
    """Re-invoke this script once per fold, each in a fresh subprocess.

    The pattern ``kfold.py`` has always used, and the only one that works: a holdout
    can be applied to a process that has not yet imported ``realistix.splits``, and to
    no other. A tool that loops folds in-process cannot be fixed by moving the
    assignment earlier, because the second fold is already too late for the first
    fold's import.

    Children inherit the parent's environment with the holdout replaced, and are
    invoked with ``flag <name>`` appended so they take the single-fold branch. The
    first non-zero exit aborts the run rather than leaving a half-filled cache set.
    """
    import subprocess

    argv = list(sys.argv[1:] if base_argv is None else base_argv)
    for name in folds:
        env = dict(os.environ)
        env.update(fold_settings(name))
        cmd = [sys.executable, sys.argv[0], *argv, flag, name, *extra]
        print(f"[foldenv] {name}: {' '.join(cmd[1:])}", flush=True)
        r = subprocess.run(cmd, env=env)
        if r.returncode:
            raise SystemExit(f"[foldenv] fold {name} exited {r.returncode}")


if __name__ == "__main__":
    for folds in FOLD_SETS.values():
        for f in folds:
            print(f"{f['name']:>10}  {fold_settings(f['name'])}")
