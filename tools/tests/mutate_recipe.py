#!/usr/bin/env python3
"""Mutation tests for the recipe/pack checkers (tools/check_recipe.py, tools/check_pack.py).

Why. A checker that has never been shown a bad artefact proves nothing: on 2026-09-11 a
pack with a ZEROED head passed the metadata-only integrity check, and on 2026-09-16 a
student replaced by seed-0 weights with copied metadata, a genuine step-2000 snapshot
and bad teacher digests all passed build_ckpt and check_pack. This script builds each
mutant from a genuine pack and asserts the checker REJECTS it (and accepts the original).
Run it after every checker change; a mutant that passes is a bug in the checker.

    python tools/tests/mutate_recipe.py --pack submissions/k4a2/model.pth [--replay]

Every mutant is written under a temporary directory on REALISTIX_BUILD (never over the
pack); the genuine sources are never modified. `--replay` also exercises the
`--replay-student` gate (GPU, ~5 min per mutant) on the two mutants only a replay can
catch: the seed-0 substitute and the step-2000 snapshot.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, shutil, subprocess, sys, tempfile
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PY = sys.executable
CHECK = os.path.join(REPO, "tools", "check_recipe.py")
INFERENCE = None


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def run_check(pack, extra=()):
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))
    r = subprocess.run([PY, CHECK, "--pack", pack, "--inference", INFERENCE, *extra], capture_output=True, text=True, env=env, cwd=REPO)
    return r.returncode, r.stdout + r.stderr


def save(obj, path):
    torch.save(obj, path); return path


def main():
    ap = argparse.ArgumentParser()
    # The default must be a pack that MATCHES the current contract, or every mutant trips the
    # contract mismatch first and the suite reads 11/14 for a reason that has nothing to do with
    # the checker. That happened on 2026-09-23 when STUDENT --steps moved 4000 -> 12000 and this
    # default still pointed at k4a2 (a 4000-step student).
    ap.add_argument("--pack", default=os.path.join(REPO, "submissions", "dec12kfinal5b", "model.pth"))
    ap.add_argument("--replay", action="store_true")
    a = ap.parse_args()
    global INFERENCE
    INFERENCE = os.path.join(os.path.dirname(os.path.abspath(a.pack)), "submission.py")
    tmp = tempfile.mkdtemp(prefix="mutants_", dir=os.environ.get("REALISTIX_BUILD") or None)
    print(f"mutants under {tmp}")
    pk = torch.load(a.pack, map_location="cpu", weights_only=False)
    st = torch.load(pk["head_path"], map_location="cpu", weights_only=False)
    results = {}

    def expect(name, pack, want_fail=True, extra=()):
        rc, out = run_check(pack, extra)
        ok = (rc != 0) if want_fail else (rc == 0)
        results[name] = {"exit": rc, "ok": ok}
        tail = [l for l in out.splitlines() if l.strip().startswith("***") or l.startswith("OK") or l.startswith("FAILED")]
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: exit {rc}" + ("" if ok else "  <-- checker did not behave as required"))
        for l in tail[:4]:
            print("      " + l.strip()[:140])

    # 0. positive control: the genuine pack must pass.
    expect("genuine pack passes", a.pack, want_fail=False)

    def mutant_pack(name, mutate_pack=None, mutate_student=None):
        """Copy the pack (and optionally a re-saved student) into tmp, apply the mutation."""
        d = os.path.join(tmp, name); os.makedirs(d)
        p2 = copy.deepcopy(pk)
        if mutate_student is not None:
            s2 = copy.deepcopy(st); mutate_student(s2)
            spath = save(s2, os.path.join(d, "student.pt"))
            p2["head_path"] = spath; p2["head_sha256"] = sha256(spath)
            p2["head"] = {k: (v.half() if v.is_floating_point() else v) for k, v in s2["head"].items()}
        if mutate_pack is not None:
            mutate_pack(p2)
        return save(p2, os.path.join(d, "model.pth"))

    # 1. zeroed head in the pack only (source untouched): tensor gate.
    expect("zeroed packed head", mutant_pack("zero_head", mutate_pack=lambda p: [v.zero_() for v in p["head"].values()]))
    # 2. head key removed from the pack.
    def drop_key(p): p["head"].pop(next(iter(p["head"])))
    expect("missing head tensor", mutant_pack("drop_key", mutate_pack=drop_key))
    # 3. one member bit flipped in the pack.
    def flip(p):
        k = next(iter(p["models"][0])); t = p["models"][0][k]; t.view(-1)[0] = t.view(-1)[0] + 1
    expect("flipped member tensor", mutant_pack("flip_member", mutate_pack=flip))
    # 4. wrong spread metadata.
    expect("wrong head_spread metadata", mutant_pack("spread_meta", mutate_pack=lambda p: p.__setitem__("head_spread", False)))
    # 5. student SOURCE replaced by teacher-0 weights, metadata copied (packing fidelity passes).
    t0 = torch.load(st["teachers"][0], map_location="cpu", weights_only=False)
    def sub_seed0(s): s["head"] = t0["head"]
    p_sub = mutant_pack("student_is_teacher0", mutate_student=sub_seed0)
    expect("student source = teacher-0 weights (metadata only)", p_sub, want_fail=not a.replay and False, extra=())
    results["student source = teacher-0 weights (metadata only)"]["note"] = ("expected to PASS metadata checks; only --replay-student catches it")
    if a.replay:
        expect("student source = teacher-0 weights (+replay)", p_sub, extra=("--replay-student",))
    # 6. student metadata claims step 2000 (an honest snapshot).
    expect("student records step 2000", mutant_pack("step2000", mutate_student=lambda s: s.__setitem__("step", 2000)))
    # 7. dishonest snapshot: step copied as 4000 but weights perturbed (only replay catches).
    def perturb(s):
        k = next(iter(s["head"])); s["head"][k] = s["head"][k] + 1e-3
    p_pert = mutant_pack("perturbed_student", mutate_student=perturb)
    expect("student weights perturbed, metadata copied", p_pert, want_fail=False)
    results["student weights perturbed, metadata copied"]["note"] = "expected to PASS metadata checks; only --replay-student catches it"
    if a.replay:
        expect("student weights perturbed (+replay)", p_pert, extra=("--replay-student",))
    # 8. bad teacher digest.
    def bad_digest(s): s["teacher_sha256"] = ["0" * 64] + list(s["teacher_sha256"][1:])
    expect("bad teacher digest", mutant_pack("bad_digest", mutate_student=bad_digest))
    # 9. duplicated teacher.
    def dup(s):
        s["teachers"] = [s["teachers"][0]] * 2 + list(s["teachers"][2:])
        s["teacher_sha256"] = [s["teacher_sha256"][0]] * 2 + list(s["teacher_sha256"][2:])
    expect("duplicated teacher", mutant_pack("dup_teacher", mutate_student=dup))
    # 10. five teachers instead of six.
    def five(s):
        s["teachers"] = list(s["teachers"][:5]); s["teacher_sha256"] = list(s["teacher_sha256"][:5])
    expect("five teachers", mutant_pack("five_teachers", mutate_student=five))
    # 11. student trained with under_weight 1 (a recipe flag).
    expect("student under_weight 1.0", mutant_pack("uw1", mutate_student=lambda s: s["args"].__setitem__("under_weight", 1.0)))
    # 12. student on other members (hash).
    def other_members(s): s["member_sha256"] = ["1" * 64] * len(s["member_sha256"])
    expect("student records other members", mutant_pack("other_members", mutate_student=other_members))
    # 13. a plain single head shipped while the script builds a student.
    def plain(s):
        for k in ("teachers", "teacher_sha256", "teacher_seeds"): s.pop(k, None)
        s["args"] = dict(t0["args"])
    expect("plain head where a student is required", mutant_pack("plain_head", mutate_student=plain))

    wrong_source = os.path.join(tmp, "wrong_submission.py")
    src = open(INFERENCE).read()
    assert "GAIN_DOSE = 0.5" in src
    open(wrong_source, "w").write(src.replace("GAIN_DOSE = 0.5", "GAIN_DOSE = 1.0", 1))
    expect("changed inference gain", a.pack, extra=("--inference", wrong_source))

    # Kernel images (2026-09-24): fused-kernel sources ship cubins that must be a fresh NVRTC compile of the source
    # embedded in submission.py. Each mutant is the genuine inference file (so source identity still holds) in its own
    # directory with its cubins altered; a non-fused source gets the one mutant that applies to it.
    import glob as _glob
    kdir = os.path.dirname(INFERENCE)
    cubins = sorted(_glob.glob(os.path.join(kdir, "fastgn_*.cubin")))

    def kernel_mutant(name, edit):
        d = os.path.join(tmp, "kern_" + name)
        os.makedirs(d)
        shutil.copy2(INFERENCE, os.path.join(d, "submission.py"))
        for c in cubins:
            shutil.copy2(c, os.path.join(d, os.path.basename(c)))
        edit(d)
        return os.path.join(d, "submission.py")

    if cubins:
        def corrupt(d):
            p = os.path.join(d, os.path.basename(cubins[0]))
            b = bytearray(open(p, "rb").read()); b[len(b) // 2] ^= 0xFF; open(p, "wb").write(bytes(b))
        expect("corrupted kernel cubin", a.pack, extra=("--inference", kernel_mutant("corrupt", corrupt)))
        expect("missing kernel cubin", a.pack,
               extra=("--inference", kernel_mutant("missing", lambda d: os.remove(os.path.join(d, os.path.basename(cubins[0]))))))
        def stale(d):
            p = os.path.join(d, os.path.basename(cubins[0]))
            os.rename(p, p.replace("fastgn_", "fastgn_000000000000_old_"))
        expect("cubin named for another kernel source", a.pack, extra=("--inference", kernel_mutant("stale", stale)))
    else:
        def stray(d):
            open(os.path.join(d, "fastgn_0123456789ab_sm80.cubin"), "wb").write(b"\x7fELF not a kernel")
        expect("stray cubin beside a kernel-free source", a.pack, extra=("--inference", kernel_mutant("stray", stray)))

    json.dump(results, open(os.path.join(tmp, "results.json"), "w"), indent=1)
    n_bad = sum(1 for r in results.values() if not r["ok"])
    print(f"\n{len(results) - n_bad}/{len(results)} mutants behaved as required; results in {tmp}/results.json")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
