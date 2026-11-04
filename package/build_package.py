#!/usr/bin/env python3
"""Assemble the Decision-Phase training package (Dockerfile, solution.py, code, README.md, METHOD.md, training logs).

    python package/build_package.py OUT_DIR [--commit REV | --worktree] [--zip ZIP]

Code files are taken from git at REV (default HEAD), so the package is exactly a committed tree; --worktree takes them
from the working tree instead (rehearsals only). The code set is the closure tools/train_final.sh needs (traced
2026-10-01): the trainers, the packing and checking tools, the realistix modules they import, the official scorer, and
the inference source of submission 947208. The package carries no weights, no data and no run artefacts: every
checkpoint the method uses is produced by train(). The build fails if any *.pth, *.pt, *.h5, *.npz, *.cubin or
token-like string is found inside.
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = [
    "tools/train_final.sh", "tools/data_manifest.py", "tools/release_manifest.json", "tools/build_cubins.py",
    "tools/train_mixed.py", "tools/train_sim2real.py", "tools/train_final_head.py", "tools/check_head_frames.py",
    "tools/distil_head.py", "tools/build_ckpt.py", "tools/pack_submission.py", "tools/build_eval_npz.py",
    "tools/check_pack.py", "tools/check_recipe.py",
    "src/realistix/__init__.py", "src/realistix/splits.py", "src/realistix/data.py", "src/realistix/normalize.py",
    "src/realistix/gpudata.py", "src/realistix/evaluate.py", "src/realistix/losses.py", "src/realistix/muon.py",
    "src/realistix/models/__init__.py", "src/realistix/models/unet.py", "src/realistix/models/uncertainty.py",
    "src/realistix/models/combiner.py",
    "starting_kit/scoring.py",
    "submissions/dec12kfinal5b/submission.py",
]
PACKAGE = ["package/Dockerfile", "package/.dockerignore", "package/solution.py", "package/README.md",
           "package/METHOD.md", "package/THIRD_PARTY_NOTICES.md"]
#: The training run behind submission 947208 (pq job 1790145274821-dec12k_rebuild, 2026-09-23, one A100-PCIE-40GB).
RUN = "runs_dec12k"
LOGS = {
    "logs/947208_training_run/dec12k_rebuild.out": os.path.expanduser("~/q/spool/logs/1790145274821-dec12k_rebuild.out"),
    "logs/947208_training_run/dec12k_rebuild.rc": os.path.expanduser("~/q/spool/logs/1790145274821-dec12k_rebuild.rc"),
    "logs/947208_training_run/dec12k_rebuild.meta.json": os.path.expanduser("~/q/spool/done/1790145274821-dec12k_rebuild.meta"),
    "logs/947208_training_run/rebuild_decision.sh": os.path.join(REPO, "research/push_2026-09-22/rebuild_decision.sh"),
}
for i in range(4):
    for f in ("args.json", "history.json"):
        LOGS[f"logs/947208_training_run/runs/dec12k_m{i}/{f}"] = f"{RUN}/dec12k_m{i}/{f}"
for s in (0, 11, 21, 31, 41, 51):
    for f in ("history.json", "lam_scan.json"):
        LOGS[f"logs/947208_training_run/runs/dec12k_head_s{s}/{f}"] = f"{RUN}/dec12k_head_s{s}/{f}"
LOGS["logs/947208_training_run/runs/dec12k_student/history.json"] = f"{RUN}/dec12k_student/history.json"

FORBIDDEN_EXT = (".pth", ".pt", ".h5", ".npz", ".npy", ".cubin", ".job")
SECRET = re.compile(r"(sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|xox[bp]-|Bearer\s+[A-Za-z0-9._-]{20,}|"
                    r"BEGIN [A-Z ]*PRIVATE KEY)")
#: Personal addresses must not leak through the attached logs (the README's contact line is deliberate).
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z.]{2,}")


def git_bytes(rev, path):
    return subprocess.run(["git", "-C", REPO, "show", f"{rev}:{path}"], check=True, capture_output=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--commit", default="HEAD")
    g.add_argument("--worktree", action="store_true")
    ap.add_argument("--zip")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        sys.exit(f"{out} exists; give a fresh directory")
    rev = None if a.worktree else subprocess.run(["git", "-C", REPO, "rev-parse", a.commit], check=True,
                                                 capture_output=True, text=True).stdout.strip()
    for rel in CODE + PACKAGE:
        data = open(os.path.join(REPO, rel), "rb").read() if rev is None else git_bytes(rev, rel)
        dst = os.path.join(out, rel[len("package/"):] if rel.startswith("package/") else rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "wb").write(data)
        if rel.endswith(".sh"):
            os.chmod(dst, 0o755)
    for rel, src in LOGS.items():
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)                 # keep the files' times: the README cites them
    open(os.path.join(out, "PACKAGE_SOURCE.txt"), "w").write(
        f"code from git {'working tree (rehearsal)' if rev is None else rev}\n")

    bad = []
    files = sorted(os.path.relpath(os.path.join(d, f), out) for d, _, fs in os.walk(out) for f in fs)
    for rel in files:
        if rel.endswith(FORBIDDEN_EXT):
            bad.append(f"forbidden file type: {rel}")
        text = open(os.path.join(out, rel), "rb").read().decode("utf-8", "replace")
        for m in SECRET.finditer(text):
            bad.append(f"secret-like string in {rel}: {m.group(0)[:12]}...")
        if rel.startswith("logs/"):
            for m in EMAIL.finditer(text):
                bad.append(f"e-mail address in {rel}: {m.group(0)[:6]}...")
    sha = hashlib.sha256(open(os.path.join(out, "submissions/dec12kfinal5b/submission.py"), "rb").read()).hexdigest()
    if sha != "1af16254833af11d3a7dfa02857801bf5b46aca380c3c7b726d735a8fcec1612":
        bad.append(f"submissions/dec12kfinal5b/submission.py is not 947208's file ({sha[:12]})")
    if bad:
        shutil.rmtree(out)
        sys.exit("refusing to build the package:\n  " + "\n  ".join(bad))
    total = sum(os.path.getsize(os.path.join(out, f)) for f in files)
    print(f"package {out}: {len(files)} files, {total / 1e6:.2f} MB, code from {rev or 'working tree'}")
    if a.zip:
        with zipfile.ZipFile(a.zip, "w", zipfile.ZIP_DEFLATED) as z:
            for rel in files:
                z.write(os.path.join(out, rel), os.path.join("realpde_t1_iapetos1918", rel))
        print(f"zip {a.zip}: {hashlib.sha256(open(a.zip, 'rb').read()).hexdigest()}")


if __name__ == "__main__":
    main()
