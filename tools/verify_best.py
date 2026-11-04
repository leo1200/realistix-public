#!/usr/bin/env python3
"""Prove every pinned submission is still intact, on both copies.

923184 scored 81.80564 and is what holds our place on the development leaderboard (918211 before it);
887158 is the reference every board delta in the documents is measured against.
Nothing in the repository protected either: the weights are gitignored, the packed
zip lived in a single directory under `build/`, and `train_final.sh` writes
`submissions/<SUB>/model.pth` with `SUB` defaulting to `all82_spread` -- so the
documented hazard in `REPRODUCE.md` was that a routine retraining run silently
replaces an artefact that is holding the shortlist place.

There is a second way to lose it that is easier than that one. The source runs live
in `runs/`, which is a directory this project deletes from with wildcards; the night
of 2026-08-17 alone removed 67 directories matching `os_t*` in two separate cleanups.
`clean_s0..2`, `hd_clean` and every later build's run directories sit in that same
tree.

So each artefact is archived twice, on two filesystems, read-only, with checksums --
and this script is the part that makes it a claim you can check rather than one you
have to believe.

    python tools/verify_best.py

Exit status is 0 only if every file on both copies matches the recorded digest.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

#: Every pinned artefact, newest first. 926634 holds the board place and 929146 is the frozen
#: method (2026-09-17); 887158 stays pinned because it is the reference every board
#: delta in the documents is computed against, and because a pinned artefact costs
#: nothing where a deleted one costs the shortlist.
#: (submission, description, zip, submission.py path at its packing commit).
#: 896716's submission.py is byte-identical to 887158's apart from comment wording
#: updated in the 2026-08-21 audit, which is why the two provenance commits differ.
#: A provenance of the form WORKTREE:<path> compares against the working tree instead of a
#: git blob: the k4g/k4a2/k4h2g05 inference files were written after the last commit. Commit them
#: and replace these with <commit>:<path> -- a working-tree comparison is a weaker pin.
PINNED = [
    ("947208", "final_score 81.99600, 2026-09-27 (retained best, rank 5: dec12kfinal5b = the contract's dec12kfinal5 + cuDNN benchmark-mode algorithm selection)",
     "dec12kfinal5b.zip", "2c4a765:submissions/dec12kfinal5b/submission.py"),
    ("943854", "final_score 81.94269, 2026-09-25 (retained best until 947208: dec12k's model with the fused-kernel inference dec12kgnv3leanb; the contract ships dec12kfinal5b, 947208's file: the same computation plus the glue kernels, the overlapped copy and benchmark mode)",
     "dec12kgnv3leanb.zip", "c8eddac:submissions/dec12kgnv3leanb/submission.py"),
    ("942498", "final_score 81.90992, 2026-09-24 (retained best: the Decision-Phase deliverable dec12k -- the a49588c/d490ad1 rebuild, 12k student -- plus stderr logging)",
     "dec12klog.zip", "d490ad1:submissions/dec12klog/submission.py"),
    ("934147", "final_score 81.90253, 2026-09-20 (board best: 932643's weights and code returning float32 arrays; a clock-lottery duplicate)",
     "canon0918f32.zip", "WORKTREE:submissions/canon0918f32/submission.py"),
    ("932643", "final_score 81.89314, 2026-09-19 (board best: the frozen method rebuilt from scratch on an A100, seeds 0-3 -- the Decision-Phase artefact)",
     "canon0918.zip", "WORKTREE:submissions/canon0918/submission.py"),   # canon0918 ships k4a2's inference source byte for byte
    ("929146", "final_score 81.78851, 2026-09-17 (the frozen method: four members, gain 0.5, distilled student head)",
     "k4a2.zip", "WORKTREE:submissions/k4a2/submission.py"),   # k4a2 ships k4g's inference source byte for byte
    ("926634", "final_score 81.80680, 2026-09-16 (board best: two output-averaged heads on the gained forecast)",
     "k4h2g05.zip", "WORKTREE:submissions/k4h2g05/submission.py"),
    ("923184", "final_score 81.80564, 2026-09-12 (four members, single head, no gain; the 09-18 clock control)", "k4.zip",
     "2568cb8:submissions/wd03mv/submission.py"),   # k4 ships wd03mv's inference source byte for byte
    ("918211", "final_score 81.76779, 2026-09-08", "wd03mv.zip",
     "2568cb8:submissions/wd03mv/submission.py"),
    ("904950", "final_score 81.59694, 2026-08-28", "tke1em8ff.zip",
     "8fe7801:submissions/tke1em8ff/submission.py"),
    ("902375", "final_score 81.51651, 2026-08-26", "spsw7k.zip",
     "9481c62:submissions/spsw7k/submission.py"),
    ("901812", "final_score 81.44630, 2026-08-25", "spsw10.zip",
     "f88cc49:submissions/spsw10/submission.py"),
    ("898763", "final_score 81.38978, 2026-08-24", "muonens.zip",
     "49e14a9:submissions/muonens/submission.py"),
    ("897732", "final_score 81.31545, 2026-08-23", "head4kv2.zip",
     "43e2055:submissions/head4kv2/submission.py"),
    ("896716", "final_score 81.24771, 2026-08-22", "head4k.zip",
     "e5763a3:submissions/head4k/submission.py"),
    ("887158", "final_score 81.15749, 2026-08-13", "all82_spread.zip",
     "c1db5a9:submissions/all82_spread/submission.py"),
]
ARCHIVE = "archive"
SECOND = os.path.expanduser("~/realistix-artefacts")
SHIPPED_COMMIT = "c1db5a9"


def digest(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sums(root: str) -> dict:
    p = os.path.join(root, "SHA256SUMS")
    if not os.path.exists(p):
        return {}
    out = {}
    for line in open(p):
        d, _, name = line.strip().partition("  ")
        out[name.lstrip("./")] = d
    return out


def check(root: str, only_zip: bool = False) -> tuple[int, int, list]:
    want, ok, bad = sums(root), 0, []
    if not want:
        return 0, 0, [f"{root}: no SHA256SUMS"]
    for name, d in sorted(want.items()):
        p = os.path.join(root, name)
        if only_zip and not name.endswith(".zip"):
            continue                      # the second copy carries the zip alone
        if not os.path.exists(p):
            bad.append(f"MISSING  {name}")
            continue
        if digest(p) != d:
            bad.append(f"CORRUPT  {name}")
        else:
            ok += 1
    return ok, len(want), bad


def main() -> None:
    fail = []
    for sub, desc, _zip, _prov in PINNED:
        print(f"submission {sub} -- {desc}\n")
        for root, label, zip_only in ((os.path.join(ARCHIVE, sub), "archive     ", False),
                                      (os.path.join(SECOND, sub), "second copy ", True)):
            if not os.path.isdir(root):
                print(f"{label} {root}\n  *** DIRECTORY MISSING ***")
                fail.append(root)
                continue
            ok, total, bad = check(root, zip_only)
            # Read-only is part of the pin, not decoration: a writable archive is one
            # `cp` away from holding a different artefact under the same name, and
            # the checksum check below only catches that after the fact. Both copies
            # were made read-only on 2026-09-11; a writable one fails from then on.
            writable = os.access(root, os.W_OK) or any(
                os.access(os.path.join(d, f), os.W_OK)
                for d, dirs, files in os.walk(root) for f in dirs + files)
            print(f"{label} {root}")
            print(f"  {ok} file(s) verified" + (f" of {total} recorded" if not zip_only else "")
                  + ("   *** WRITABLE -- the pin requires read-only ***" if writable else "   [read-only]"))
            if writable:
                fail.append(f"writable {root}")
            for b in bad:
                print(f"  {b}")
            fail += bad
        print()

    # The artefact is only meaningful next to the code that ran it: each zip's
    # submission.py is compared against the file at its own packing commit.
    try:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for sub, _desc, zname, prov in PINNED:
            z = subprocess.run(["unzip", "-p", os.path.join(ARCHIVE, sub, zname),
                                "submission.py"], capture_output=True)
            if prov.startswith("WORKTREE:"):
                wt = os.path.join(repo, prov.split(":", 1)[1])
                blob = open(wt, "rb").read() if os.path.exists(wt) else b""
                same = z.returncode == 0 and bool(blob) and z.stdout == blob
            else:
                g = subprocess.run(["git", "show", prov], capture_output=True, cwd=repo)
                same = z.stdout == g.stdout and z.returncode == 0 and g.returncode == 0
            print(f"provenance   {sub} zip's submission.py vs {prov.split(':')[0]}: "
                  + ("identical" if same else "*** DIFFERS ***"))
            if not same:
                fail.append(f"provenance {sub}")
    except FileNotFoundError as e:
        # `unzip` or `git` missing is not a pass: an unverifiable provenance is a failure
        # of the check, and it used to print a note and exit 0.
        print(f"\nprovenance   *** could not check ({e}) ***")
        fail.append("provenance tools")

    print("\n" + (f"OK -- all {len(PINNED)} pinned submissions are intact on both "
                   "copies." if not fail
                   else f"FAILED -- {len(fail)} problem(s) above."))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
