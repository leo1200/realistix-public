#!/usr/bin/env python3
"""Assemble a submission zip: authored code + kit deps + checkpoint.

Only ``submission.py`` (and anything else we actually wrote) lives in the repo.
The vendored kit code and the 201 MB checkpoint are copied in at pack time, so
git never sees them. The staging tree and the zip go under <data-disk> because
$HOME is nearly full.

    python tools/pack_submission.py fno_band --ckpt sim_real_fno_fp16.pth
    python tools/pack_submission.py --audit            # who shares submission.py

**On the duplication.** Every scored candidate keeps its own copy of
``submission.py``, because the archive is meant to be a record of what was actually
uploaded and a submission has to be self-contained. The cost is that a correctness
fix to ``predict()`` needs one edit per copy and nothing notices if you miss one --
which happened: the docstring claimed training on "all 82 real trajectories" in nine
directories at once, eight days after ``7575_0`` was withdrawn. ``--audit`` groups
the directories by content hash so the blast radius of an edit is visible before you
make it. Fix the shipped one first, then propagate to everything in its group.
"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
KIT = os.path.join(REPO, "starting_kit")
BUILD = os.environ.get("REALISTIX_BUILD", "build")
CKPT_ROOT = os.environ.get(
    "REALPDE_CKPTS",
    "data/RealPDE-Competition-Data/baseline_checkpoints")
SIZE_CAP = 256 * 1024 * 1024
# The submission directory currently on the board (923184, k4). Kept in one
# place so `--audit` and the docs cannot drift apart again.
SHIPPED = os.environ.get("REALISTIX_SHIPPED", "k4")

# Kit pieces a submission that wraps a baseline needs at the archive root.
KIT_DEPS = ["load_baseline.py", "rpde_baselines", "_vendor"]


def _check_links(path, root):
    """Reject any symlink under `path` that escapes `root`.

    copytree and copy2 both dereference by default, so a link anywhere in the tree
    pulls outside data into the uploaded archive as a regular file. Checking only the
    entries os.listdir returns caught the top level and missed `nested/leak`.
    """
    real_root = os.path.realpath(root)
    def bad(p):
        return os.path.islink(p) and not (
            os.path.realpath(p) == real_root
            or os.path.realpath(p).startswith(real_root + os.sep))
    if bad(path):
        raise SystemExit(f"{path} is a symlink pointing outside {root} "
                         f"({os.path.realpath(path)}); refusing to package it")
    for r, dirs, files in os.walk(path):
        for name in dirs + files:
            p = os.path.join(r, name)
            if bad(p):
                raise SystemExit(f"{p} is a symlink pointing outside {root} "
                                 f"({os.path.realpath(p)}); refusing to package it")


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_ckpt(name):
    for sub in ("sim_real_ft", "sim_pretrain"):
        p = os.path.join(CKPT_ROOT, sub, name)
        if os.path.exists(p):
            return p
    raise SystemExit(f"checkpoint {name} not found under {CKPT_ROOT}")


def audit() -> None:
    """Group submissions/*/submission.py by content, so shared copies are visible."""
    import hashlib
    root = os.path.join(REPO, "submissions")
    groups: dict[str, list[str]] = {}
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "submission.py")
        if not os.path.exists(p):
            continue
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]
        groups.setdefault(h, []).append(name)

    # The current champion, not the historical default build target. This said
    # `all82_spread` until 2026-09-01, which is the 887158-era artefact; the audit
    # therefore marked the wrong group as shipped.
    shipped = os.path.join(root, SHIPPED, "submission.py")
    live = (hashlib.sha256(open(shipped, "rb").read()).hexdigest()[:12]
            if os.path.exists(shipped) else None)
    for h, names in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        mark = "  <- the shipped file" if h == live else ""
        n = len(names)
        print(f"{h}  {n} cop{'y' if n == 1 else 'ies'}{mark}")
        print("            " + "  ".join(names))
    print(f"\n{len(groups)} distinct versions across "
          f"{sum(len(v) for v in groups.values())} directories. An edit to the "
          f"shipped file\nneeds propagating to the rest of its group by hand.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", help="directory under submissions/")
    ap.add_argument("--audit", action="store_true",
                    help="report which submission.py copies are identical, and exit")
    ap.add_argument("--ckpt", default=None, help="baseline checkpoint filename to bundle")
    ap.add_argument("--stale-ok", action="store_true",
                    help="package even if model.pth predates the head it names")
    ap.add_argument("--no-kit-deps", action="store_true",
                    help="skip load_baseline/rpde_baselines/_vendor (self-contained model)")
    args = ap.parse_args()

    if args.audit:
        audit()
        return
    if not args.name:
        ap.error("name is required unless --audit is given")

    # `name` indexes a directory under submissions/, and it is about to be joined
    # into a path this script then deletes recursively. os.path.join discards its
    # left operand for an absolute right one, so `pack_submission.py /some/dir`
    # made both src and stage that directory -- passing the isdir check and then
    # rmtree-ing it. A bare basename is the only thing that can be meant here.
    if args.name != os.path.basename(args.name) or args.name in ("", ".", ".."):
        raise SystemExit(f"name must be a bare directory name under submissions/, "
                         f"not a path: {args.name!r}")

    src = os.path.join(REPO, "submissions", args.name)
    if not os.path.isdir(src):
        raise SystemExit(f"no such submission: {src}")

    stage = os.path.join(BUILD, args.name)
    if os.path.realpath(stage) == os.path.realpath(src):
        raise SystemExit(f"staging path is the source path ({src}); refusing to delete it")
    if not os.path.realpath(stage).startswith(os.path.realpath(BUILD) + os.sep):
        raise SystemExit(f"staging path escapes {BUILD}: {stage}")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)

    for entry in sorted(os.listdir(src)):
        s = os.path.join(src, entry)
        # Follow no symlinks out of the submission tree: copytree/copy2 dereference
        # by default, so a link inside a submission directory would pull arbitrary
        # files from elsewhere on the machine into the uploaded archive.
        _check_links(s, src)
        (shutil.copytree if os.path.isdir(s) else shutil.copy2)(s, os.path.join(stage, entry))

    if not args.no_kit_deps:
        for dep in KIT_DEPS:
            s = os.path.join(KIT, dep)
            (shutil.copytree if os.path.isdir(s) else shutil.copy2)(s, os.path.join(stage, dep))

    if args.ckpt:
        shutil.copy2(find_ckpt(args.ckpt), os.path.join(stage, args.ckpt))

    for root, dirs, _ in os.walk(stage):
        for d in list(dirs):
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d))
                dirs.remove(d)

    if not os.path.exists(os.path.join(stage, "submission.py")):
        raise SystemExit("submission.py must be at the archive root")

    # Refuse to package weights that were not built from the head they claim. When
    # build_ckpt failed on 2026-08-12 this script repacked the previous model.pth and
    # reported success, emitting a zip with the old checksum under a description
    # saying otherwise -- a pipeline that fails *forward* into a plausible artefact is
    # worse than one that stops. Comparing against submission.py's mtime instead was
    # the obvious check and the wrong one: editing the code legitimately makes the
    # weights older, and it false-positived within a day.
    ck = os.path.join(stage, "model.pth")
    if os.path.exists(ck) and not args.stale_ok:
        import torch
        meta = torch.load(ck, map_location="cpu")
        hp, hm = meta.get("head_path"), meta.get("head_mtime")
        if hp and os.path.exists(hp) and abs(os.path.getmtime(hp) - hm) > 1.0:
            raise SystemExit(f"model.pth was built from an older {hp}; "
                             f"rerun tools/build_ckpt.py or pass --stale-ok")
        # The guard above validates nothing when the recorded head is gone -- moved,
        # deleted, or simply on another machine -- and mtime was never an identity
        # check anyway. Say so rather than passing silently, and verify the member
        # hashes recorded by build_ckpt where they exist.
        if not hp:
            print("warning: model.pth records no head provenance (built before "
                  "2026-08-17); nothing to validate")
        elif not os.path.exists(hp):
            print(f"warning: recorded head {hp} is not on this machine; "
                  f"provenance is unverified")
        mp, mh = meta.get("member_paths", []), meta.get("member_sha256", [])
        if len(mp) != len(mh):
            raise SystemExit(f"model.pth records {len(mp)} member paths but "
                             f"{len(mh)} hashes; rebuild it")
        for p, want in zip(mp, mh):
            if not os.path.exists(p):
                print(f"warning: member {p} is not on this machine; unverified")
            elif _sha256(p) != want:
                raise SystemExit(f"member {p} no longer hashes to what model.pth "
                                 f"records; rebuild or pass --stale-ok")

    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(stage) for f in fs)
    if total > SIZE_CAP:
        raise SystemExit(f"extracted size {total/1e6:.1f} MB exceeds the 256 MB cap")

    zip_path = os.path.join(BUILD, f"{args.name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(stage):
            for f in files:
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, stage))

    print(f"staged  {stage}")
    print(f"extracted {total/1e6:.1f} MB of {SIZE_CAP/1e6:.0f} MB cap")
    print(f"zip     {zip_path}  ({os.path.getsize(zip_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
