#!/usr/bin/env python3
"""Prove `tools/train_final.sh` still builds the model we actually ship.

Why this exists. On 2026-08-24 the script was four changes behind the submission
holding our leaderboard place: the head step (896716, +0.414 board sps), the head
lr and tau_frac (897732, +0.320), and the Muon optimizer with its end-of-cosine
stopping step (898763, +0.074 of final). Running it would have reproduced the
887158-era model -- about 0.23 of final below the pinned best -- and nothing in the
repository would have said so. On 2026-09-10 it happened again: the defaults built
a 904950-era model and reproducing 918211 needed three environment overrides. That
matters more than a stale comment usually does, because `train_final.sh` is not a
convenience script: the top 10 are re-trained from scratch by the organizers before
the prizes are decided, so it *is* the Decision Phase submission. A recipe that
drifts from the artefact is a silent loss in the one phase that pays.

What it proves (2026-09-11 scope; before that it read one member and a regex over
the script's text):

  1. The *resolved* recipe. `PRINT_RECIPE=1 bash tools/train_final.sh` makes the
     script print the argument lists it would pass to the two trainers, literal
     plus every override in effect in the caller's environment, and exit. Those
     lists are fed through the trainers' own `build_parser()`, so the guard cannot
     drift from the flag definitions the way a reimplemented copy would.
  2. Every packed member and the head. The shipped `model.pth` records the member
     and head checkpoint paths and their SHA256 at packing time (build_ckpt). Each
     is re-hashed on disk (a run directory overwritten since packing fails here)
     and its recorded `args` compared with the resolved recipe.
  3. The head was calibrated on exactly the packed members (paths, and -- for heads
     fitted since 2026-09-11 -- the members' digests recorded at fit time).
  3b. The packed TENSORS are the source checkpoints' tensors: every member is
     re-folded (build_ckpt.fold_modality) and cast to fp16 from its checkpoint and
     compared bit-for-bit with the pack, the head likewise, and the architecture
     fields the submission rebuilds modules from are compared with the recorded
     args. Provenance metadata alone let a pack with a zeroed head pass (Codex,
     2026-09-11 round 2).
  4. Data provenance: with the research holdout variables cleared, `realistix.splits`
     must see 81 real (7575_0 excluded) and 100 simulated trajectories.
  5. The caller's environment: any REALISTIX_* variable that would change the
     training set is reported (the script clears them, so this is a warning).

Three kinds of flag disagreement are reported and all of them fail:

  set differently  the script names a flag and the checkpoint disagrees
  script-only      the script sets something the shipped run did not
  run-only         the shipped run differs from the parser default on a flag the
                   script never names -- the drift mode that bit us, because a new
                   flag added to a run is invisible to a script that predates it

A checkpoint that has no record of a flag at all was trained before that flag
existed. That is reported as `postdates` and does not fail, but only while the
script leaves the flag at its default -- if the script sets it, the shipped run
cannot have honoured it and the disagreement is real.

    python tools/check_recipe.py                       # against the pinned best's pack
    python tools/check_recipe.py --pack submissions/rebuild/model.pth
    python tools/check_recipe.py --member RUN --head RUN   # explicit checkpoints
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import shlex
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

import build_ckpt  # noqa: E402
import distil_head  # noqa: E402
import train_final_head  # noqa: E402
import train_mixed  # noqa: E402

SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def verify_kernel_images(inference: str, bad: list) -> None:
    """Fused-kernel inference (dec12kgn*, 2026-09-24) ships cubins compiled from the CUDA source embedded in
    submission.py; they must be byte-for-byte a fresh NVRTC compile of that source (tools/build_cubins.py), and a
    source without kernels must ship none. Inert for a non-fused source with no cubins beside it."""
    sys.path.insert(0, HERE)
    import build_cubins as bc
    d = os.path.dirname(os.path.abspath(inference))
    src = bc.kernel_source(inference) if os.path.isfile(inference) else None
    shipped = sorted(glob.glob(os.path.join(d, "fastgn_*.cubin")))
    print(f"\n=== kernel images beside {inference}")
    if src is None:
        if shipped:
            bad.append("kernel images")
            print(f"  *** no kernel source in the inference file, yet {len(shipped)} fastgn cubin(s) beside it")
        else:
            print("  no fused-kernel source, no cubins: nothing to verify")
        return
    h = hashlib.sha256(src.encode()).hexdigest()[:12]
    want = {os.path.join(d, f"fastgn_{h}_sm{arch}.cubin"): bc.compile_cubin(src, arch) for arch in bc.ARCHES}
    off = [p for p in shipped if p not in want] + [p for p, img in want.items()
                                                    if not os.path.isfile(p) or open(p, "rb").read() != img]
    if off:
        bad.append("kernel images")
        for p in off:
            print(f"  *** {os.path.basename(p)}: missing, stale or not a fresh NVRTC compile of the shipped source")
    else:
        print(f"  kernel source sha12 {h}: {len(want)} cubins, each byte-identical to a fresh NVRTC compile")


def champion_source() -> str:
    """The inference source train_final.sh copies into a fresh build: its CHAMPION literal
    (or $CHAMPION, exactly as the script resolves it). The pack's submission.py must be this
    file byte for byte, so the inference contract is bound to the committed script rather than
    to a hash that goes stale when the contract changes (2026-09-24 audit: dec12k was built
    with CHAMPION=canon0918f32 while the literal said k4a2, and nothing noticed)."""
    m = re.search(r"^CHAMPION=\$\{CHAMPION:-([^}]+)\}", open(SCRIPT).read(), re.M)
    if not m:
        raise SystemExit(f"no CHAMPION literal in {SCRIPT}")
    name = os.environ.get("CHAMPION") or m.group(1)
    return os.path.join(HERE, "..", "submissions", name, "submission.py")

SCRIPT = os.path.join(HERE, "train_final.sh")

#: The pack of the frozen method: 929146 (`k4a2`, 2026-09-17) -- 923184's four members and the
#: distilled student head; its recorded provenance names the members, the student and the
#: student's six teachers. 926634 (the two-head pair) holds the board place but is not the
#: method the script rebuilds. Update this and `tools/verify_best.py` together.
SHIPPED_PACK = os.path.join(REPO, "submissions", "dec12kfinal5b", "model.pth")   # dec12k's model (the a49588c rebuild, 12k student) with the fused-kernel inference (943854, 2026-09-25)

#: Flags whose value is a path, a device or a seed rather than a modelling choice.
#: They differ between any two runs by construction and say nothing about the recipe.
IGNORE = {"out", "seed", "runs", "sim_dir", "snapshots", "eval_every", "distill_unsafe",
          "REALISTIX_HELDOUT_AOA", "REALISTIX_HELDOUT_RE", "REALISTIX_HELDOUT_CELLS",
          # Added to the recorded args on 2026-09-01 so distillation can check every
          # holdout axis. Like the other three it is the run's environment, not its
          # recipe; omitting it here made this tool fail on any checkpoint trained
          # after that change while passing on every older one.
          "REALISTIX_HELDOUT_TAIL", "REALISTIX_SIM_KEEPS_CELLS"}

#: Fields that must be ABSENT/falsy rather than merely equal. These change the learned
#: function or its provenance, and the script never sets them -- so a checkpoint that
#: records one was not built by the script, whatever else matches. They used to sit in
#: IGNORE as "paths, not modelling choices", which let a member recording a leaky
#: warm start, a distillation teacher, a combiner or extra spread members pass as
#: "matches the shipped recipe on every flag".
MUST_BE_UNSET = ("init_from", "distill_from", "cooperate_with", "combiner", "spread_runs", "cpu_data",
                 "output_adapter")

#: Environment variables that change what the trainers train on. The script clears
#: them; a caller that has them set is told so, because every other tool inherits them.
DATA_ENV = ("REALISTIX_HELDOUT_AOA", "REALISTIX_HELDOUT_RE", "REALISTIX_HELDOUT_CELLS",
            "REALISTIX_HELDOUT_TAIL", "REALISTIX_SIM_KEEPS_CELLS", "REALISTIX_SIM_DIR",
            "REALISTIX_ALLOW_MIXED_HOLDOUT", "REALPDE_DATA_OVERRIDE")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolved_recipe() -> tuple[list[str], list[str], list[str], str, list[str], list[str]]:
    """(TRAIN argv, HEAD argv, member seeds, HEAD_MODE, teacher seeds, STUDENT argv) as the
    script would run them now."""
    env = dict(os.environ, PRINT_RECIPE="1")
    r = subprocess.run(["bash", SCRIPT], capture_output=True, text=True, env=env, cwd=REPO)
    if r.returncode != 0:
        raise SystemExit(f"PRINT_RECIPE=1 bash {SCRIPT} failed:\n{r.stdout}\n{r.stderr}")
    got = {}
    for line in r.stdout.splitlines():
        m = re.match(r"^RESOLVED_(TRAIN|HEAD_MODE|HEAD|MEMBERS|TEACHERS|STUDENT): ?(.*)$", line)
        if m:
            got[m.group(1)] = m.group(2)
    for k in ("TRAIN", "HEAD", "MEMBERS", "HEAD_MODE", "TEACHERS", "STUDENT"):
        if k not in got:
            raise SystemExit(f"{SCRIPT} printed no RESOLVED_{k} line:\n{r.stdout}")
    return (shlex.split(got["TRAIN"]), shlex.split(got["HEAD"]), got["MEMBERS"].split(),
            got["HEAD_MODE"].strip(), got["TEACHERS"].split(), shlex.split(got["STUDENT"]))


def literal_flags(var: str) -> list[str]:
    """The argv the script's literal would pass with every ${X:-default} at default.

    Kept next to the resolved recipe so an override in effect in the caller's
    environment is named rather than silently accepted.
    """
    src = open(SCRIPT).read()
    m = re.search(rf'^{var}="((?:[^"\\]|\\.)*)"', src, re.M | re.S)
    if not m:
        raise SystemExit(f"{SCRIPT}: no {var}=\"...\" assignment found")
    body = m.group(1).replace("\\\n", " ")
    body = re.sub(r"\$\{[A-Za-z_][A-Za-z0-9_]*:-([^}]*)\}", r"\1", body)
    if "$" in body:
        raise SystemExit(f"{var}: unexpanded shell expansion in {body!r}")
    return shlex.split(body)


def parsed(parser, argv: list[str]) -> tuple[dict, dict, set]:
    """(values, defaults, flags this argv actually names)."""
    named = set()
    for tok in argv:
        if tok.startswith("--"):
            named.add(tok[2:].split("=")[0].replace("-", "_"))
    # `--out` is required by every parser and is not part of the recipe; `--teachers` is the
    # student's required input list (paths, checked separately by hash).
    fill = ["--out", "/dev/null"]
    if any(a.dest == "teachers" for a in parser._actions):
        fill += ["--teachers", "/dev/null"]
    values = vars(parser.parse_args(argv + fill))
    defaults = {a.dest: a.default for a in parser._actions if a.dest != "help"}
    return values, defaults, named


def same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))
        except (TypeError, ValueError):
            return False
    return a == b


def compare(label: str, parser, argv: list[str], ckpt_args: dict, ignore: set | None = None) -> list[str]:
    ignore = IGNORE if ignore is None else ignore
    values, defaults, named = parsed(parser, argv)
    if "head_frames" in ckpt_args:
        # Recorded provenance derived by train_final_head, not a CLI option.
        # The script clears REALISTIX_HELDOUT_TAIL before training, so the frames a
        # build's head sees are always "all", whatever this shell carries.
        values["head_frames"] = "all"
        defaults["head_frames"] = values["head_frames"]
    bad, info = [], []
    print(f"\n{label}")
    print(f"  script: {' '.join(argv)}")
    for k in MUST_BE_UNSET:
        if ckpt_args.get(k):
            bad.append(f"  provenance       --{k.replace('_','-')}={ckpt_args[k]!r}: the "
                       f"script never sets this, so the shipped run was not built by it")
    for k in sorted(set(values) | set(ckpt_args)):
        if k in MUST_BE_UNSET or k in ignore:
            continue
        want, got = values.get(k, "<absent>"), ckpt_args.get(k, "<absent>")
        if same(want, got):
            continue
        if k not in ckpt_args:
            # The run predates the flag. Harmless exactly when the script does not
            # set it: the code that trained this checkpoint behaved as the default.
            if k not in named and same(want, defaults.get(k)):
                info.append(f"  postdates        --{k.replace('_','-')}: added to the "
                            f"parser after this run; script leaves it at {want!r}")
            else:
                bad.append(f"  set differently  --{k.replace('_','-')}: script "
                           f"{want!r}, but the shipped run predates the flag")
            continue
        if k in named:
            bad.append(f"  set differently  --{k.replace('_','-')}: "
                       f"script {want!r}, shipped run {got!r}")
        elif k in defaults and not same(defaults[k], got):
            bad.append(f"  run-only         --{k.replace('_','-')}: "
                       f"shipped run {got!r}, script leaves it at the default "
                       f"{defaults[k]!r}")
        else:
            bad.append(f"  script-only      --{k.replace('_','-')}: "
                       f"script {want!r}, shipped run {got!r}")
    for b in bad + info:
        print(b)
    if not bad:
        print("  matches the shipped run on every recipe flag")
    return bad


def load_ckpt(label: str, path: str, want_sha: str | None, bad: list) -> dict | None:
    real = os.path.realpath(path)
    if not os.path.exists(real):
        print(f"\n{label}: {path} does not resolve to a file")
        bad.append(f"{label} missing")
        return None
    if want_sha is not None:
        if not SHA_RE.match(str(want_sha)):
            print(f"\n{label}: {real}\n  *** the pack records a malformed digest {want_sha!r} for this file")
            bad.append(f"{label} digest format")
            return None
        got = sha256(real)
        if got != want_sha:
            print(f"\n{label}: {real}\n  *** SHA256 differs from the one recorded at packing"
                  f" ({got[:12]}... vs {want_sha[:12]}...) -- the run directory was rewritten"
                  f" after the pack was built; its args no longer describe the packed weights")
            bad.append(f"{label} sha256")
            return None
    ck = torch.load(real, map_location="cpu", weights_only=False)
    if "args" not in ck:
        print(f"\n{label}: {real} records no args")
        bad.append(f"{label} args")
        return None
    # A member shipped from a snapshot carries the RUN's args, in which `fixed_step`
    # is whatever the run was told to keep -- not the step this checkpoint actually
    # is. What ships is the kept step, so compare that.
    if "step" in ck and ck["args"].get("real_trajs") == "all":
        ck["args"] = dict(ck["args"], fixed_step=ck["step"])
    print(f"\n=== {label}: {real}" + ("  [sha256 matches the pack]" if want_sha else ""))
    return ck


def tensors_equal(a: dict, b: dict) -> list[str]:
    """Keys and values that differ between two state dicts (bitwise)."""
    out = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            out.append(f"{k} ({'pack-only' if k in b else 'source-only'})")
            continue
        x, y = a[k], b[k]
        if x.shape != y.shape or x.dtype != y.dtype or not torch.equal(x, y):
            out.append(k)
    return out


#: Fields submission.py indexes without a default, and the type each must have. An
#: absent one passes a `.get()`-based check and raises KeyError at inference.
PACK_SCHEMA = {"models": list, "width": int, "mults": str, "blocks": int, "anchor": str, "film": bool,
               "cond": str, "head": dict, "head_width": int, "head_spread": bool,
               "member_paths": list, "member_sha256": list, "head_path": str, "head_sha256": str}


def verify_pack_schema(pk: dict, bad: list) -> None:
    print("\n=== pack schema")
    missing = [k for k in PACK_SCHEMA if k not in pk]
    wrong = [f"{k}={type(pk[k]).__name__}" for k, t in PACK_SCHEMA.items() if k in pk and not isinstance(pk[k], t)]
    if missing or wrong:
        bad.append("pack schema")
        print(f"  *** missing {missing}, wrong type {wrong}")
    else:
        print(f"  all {len(PACK_SCHEMA)} loader-mandatory fields present with the expected types")


def verify_pack_tensors(pk: dict, member_ckpts: list[dict], head_ckpt: dict | None, bad: list) -> None:
    """The packed tensors must be exactly what build_ckpt derives from the sources."""
    print("\n=== packed tensors vs source checkpoints")
    if len(pk.get("models") or []) != len(member_ckpts):
        bad.append("pack member tensors")
        print(f"  *** the pack holds {len(pk.get('models') or [])} member state dicts for "
              f"{len(member_ckpts)} recorded members")
        return
    for i, ck in enumerate(member_ckpts):
        a = ck["args"]
        sd, _ = build_ckpt.fold_modality(build_ckpt.fold_decomposed_output(ck["model"], a))
        diff = tensors_equal(build_ckpt.half(sd), pk["models"][i])
        if diff:
            bad.append(f"member {i} tensors")
            print(f"  *** member {i}: {len(diff)} tensor(s) differ from the folded fp16 source, "
                  f"e.g. {diff[:3]}")
        else:
            print(f"  member {i}: {len(sd)} tensors identical to the folded fp16 source")
        arch = dict(width=a["width"], mults=str(a.get("mults", "1,2,4")), blocks=int(a.get("blocks", 2)),
                    anchor=a.get("anchor", "last"))
        got = dict(width=pk.get("width"), mults=str(pk.get("mults")), blocks=int(pk.get("blocks") or 2),
                   anchor=pk.get("anchor") or "last")
        if arch != got:
            bad.append(f"member {i} architecture")
            print(f"  *** member {i} architecture {arch} vs pack {got}")
    if pk.get("film") or (pk.get("cond") not in (None, "none")):
        bad.append("pack conditioning")
        print(f"  *** the pack still carries conditioning (film={pk.get('film')}, cond={pk.get('cond')}); "
              f"the shipped network is the folded plain U-Net")
    if head_ckpt is not None:
        diff = tensors_equal(build_ckpt.half(head_ckpt["head"]), pk["head"])
        if diff:
            bad.append("head tensors")
            print(f"  *** head: {len(diff)} tensor(s) differ from the fp16 source, e.g. {diff[:3]}")
        else:
            print(f"  head: {len(pk['head'])} tensors identical to the fp16 source")
        ha = head_ckpt["args"]
        if pk.get("head_width") != ha["width"] or bool(pk.get("head_spread")) != bool(ha.get("spread")):
            bad.append("head architecture")
            print(f"  *** head width/spread in the pack ({pk.get('head_width')}, {pk.get('head_spread')}) "
                  f"vs the checkpoint ({ha['width']}, {ha.get('spread')})")
        if pk.get("combiner") is not None or ha.get("combiner"):
            bad.append("combiner")
            print("  *** a combiner is packed or recorded; the shipped recipe has none")


def check_student(ck: dict, student_argv: list[str], head_argv: list[str], teacher_seeds: list[str],
                  member_shas: list[str], hp, bad_list: list) -> list[str]:
    """The distilled student and its teachers against the script's STUDENT and HEAD literals.

    Provenance chain (2026-09-17): the pack records the student's hash; the student records
    its teachers' paths, hashes and seeds and the members' hashes; each teacher records the
    members' hashes. Everything re-hashed on disk. What this cannot tell is whether the
    student file *is* the 4000-step output of those teachers rather than a snapshot or a
    substitute with copied metadata -- `--replay-student` is that gate.
    """
    bad: list[str] = []
    sp = distil_head.build_parser()
    # The student's seed is a recipe constant (7), not a run identifier: compare it.
    # `spread`/`tau_frac`/`n_models`/`head_frames` are provenance the student copies from its
    # teachers (build_ckpt reads them); the teachers themselves are compared below.
    bad += compare("interval-head student", sp, student_argv, ck["args"],
                   ignore=(IGNORE - {"seed"}) | {"teachers", "spread", "tau_frac", "n_models", "head_frames"})
    sv, _, _ = parsed(sp, student_argv)
    if ck.get("step") != sv["steps"]:
        bad.append("student step")
        print(f"  *** the student checkpoint is step {ck.get('step')}, the script trains {sv['steps']}")
    tpaths = list(ck.get("teachers") or [])
    tshas = list(ck.get("teacher_sha256") or [])
    print(f"\n=== {len(tpaths)} teacher(s) recorded by the student")
    if len(tpaths) != len(teacher_seeds):
        bad.append("teacher count")
        print(f"  *** the student was distilled from {len(tpaths)} teachers, the script fits {len(teacher_seeds)}")
    if len(tshas) != len(tpaths) or not all(SHA_RE.match(str(h)) for h in tshas):
        bad.append("teacher digests")
        print("  *** the student records no well-formed SHA256 for every teacher")
        return bad
    if len(set(tshas)) != len(tshas) or len({os.path.realpath(p) for p in tpaths}) != len(tpaths):
        bad.append("duplicate teachers")
        print("  *** a teacher appears twice (same path or same digest)")
    seen_seeds = []
    for i, (tp, sha) in enumerate(zip(tpaths, tshas)):
        tck = load_ckpt(f"teacher {i}", tp, sha, bad)
        if tck is None:
            continue
        ta = tck["args"]
        seen_seeds.append(ta.get("seed"))
        bad += compare(f"teacher {i} (seed {ta.get('seed')})", hp, head_argv, ta)
        hv, _, _ = parsed(hp, head_argv)
        if tck.get("step") is not None and hv.get("fixed_step") and tck["step"] != hv["fixed_step"]:
            bad.append(f"teacher {i} step")
            print(f"  *** teacher {i} is step {tck['step']}, the script keeps step {hv['fixed_step']}")
        if member_shas and list(tck.get("member_sha256") or []) != member_shas:
            bad.append(f"teacher {i} members")
            print(f"  *** teacher {i} was fitted on members other than the packed ones (hash)")
    if sorted(str(x) for x in seen_seeds) != sorted(teacher_seeds) and len(seen_seeds) == len(tpaths):
        bad.append("teacher seeds")
        print(f"  *** teacher seeds {seen_seeds} are not the script's TEACHER_SEEDS {teacher_seeds}")
    if member_shas and list(ck.get("member_sha256") or []) != member_shas:
        bad.append("student members")
        print("  *** the student records members other than the packed ones (hash)")
    if not bad:
        print("  student and every teacher match the script's literals; hashes chain to the packed members")
    return bad


def replay_student(student_path: str, ck: dict) -> list[str]:
    """Re-run the distillation from the recorded teachers and compare tensors bitwise."""
    import tempfile
    print("\n=== student replay (tools/distil_head.py from the recorded teachers)")
    a = ck["args"]
    tmp = tempfile.mkdtemp(prefix="replay_student_", dir=os.environ.get("REALISTIX_BUILD") or None)
    argv = [sys.executable, os.path.join(HERE, "distil_head.py"), "--teachers", ",".join(ck["teachers"]),
            "--out", tmp, "--steps", str(a["steps"]), "--batch", str(a["batch"]), "--lr", str(a["lr"]),
            "--width", str(a["width"]), "--seed", str(a["seed"]), "--train-split", str(a["train_split"]),
            "--init-teacher", str(a["init_teacher"]), "--under-weight", str(a["under_weight"])]
    env = {k: v for k, v in os.environ.items() if k not in DATA_ENV}
    r = subprocess.run(argv, capture_output=True, text=True, env=dict(env, PYTHONPATH=os.path.join(REPO, "src")), cwd=REPO)
    if r.returncode != 0:
        print("  *** replay failed:\n" + (r.stderr or r.stdout)[-2000:])
        return ["student replay run"]
    rep = torch.load(os.path.join(tmp, "best.pt"), map_location="cpu", weights_only=False)
    diff = tensors_equal(rep["head"], ck["head"])
    if diff:
        print(f"  *** {len(diff)} of {len(ck['head'])} student tensors differ from the replay, e.g. {diff[:3]} "
              f"(same card class required; a different class gives a different model)")
        return ["student replay"]
    print(f"  all {len(ck['head'])} student tensors reproduce bitwise from the recorded teachers ({tmp})")
    return []


def data_provenance(bad: list) -> None:
    """81 real + 100 sim under a cleared research environment, via the script's interpreter."""
    env = {k: v for k, v in os.environ.items() if k not in DATA_ENV}
    code = ("from realistix.splits import available, EXCLUDED; r=available('real'); s=available('sim');"
            "print(len(r), len(s), '7575_0' in {t.stem for t in r}, sorted(EXCLUDED))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=dict(env, PYTHONPATH=os.path.join(REPO, "src")), cwd=REPO)
    print("\n=== data provenance (research holdout variables cleared)")
    if r.returncode != 0:
        print(r.stderr.strip()); bad.append("data provenance"); return
    n_real, n_sim, has_dup, excluded = r.stdout.split(maxsplit=3)
    print(f"  {n_real} real, {n_sim} sim; 7575_0 present: {has_dup}; excluded: {excluded.strip()}")
    if (n_real, n_sim, has_dup) != ("81", "100", "False"):
        bad.append("data provenance")
        print("  *** expected 81 real (7575_0 excluded) and 100 sim")
    # Identity, not a count: the ordered list, sizes and digests of the release files
    # the trainers actually enumerate (tools/data_manifest.py, tools/release_manifest.json).
    r = subprocess.run([sys.executable, os.path.join(HERE, "data_manifest.py"), "--check",
                        os.path.join(HERE, "release_manifest.json")], capture_output=True,
                       text=True, env=env, cwd=REPO)
    print("  " + (r.stdout.strip() or r.stderr.strip()).replace("\n", "\n  "))
    if r.returncode != 0:
        bad.append("data manifest")
    inherited = {k: os.environ[k] for k in DATA_ENV if os.environ.get(k) not in (None, "", "0")}
    if inherited:
        print(f"  warning: this shell carries {inherited}; train_final.sh clears them, other tools do not")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=SHIPPED_PACK,
                    help="a packed model.pth whose recorded provenance names members and head")
    ap.add_argument("--inference", default="",
                    help="inference source; defaults to submission.py next to --pack")
    ap.add_argument("--member", action="append", default=[],
                    help="a forecaster checkpoint to check instead of the pack's (repeatable)")
    ap.add_argument("--head", default="", help="an interval-head checkpoint instead of the pack's")
    ap.add_argument("--replay-student", action="store_true",
                    help="re-run tools/distil_head.py from the recorded teachers (GPU, ~5 min) and "
                         "require the packed student's tensors bitwise -- the one gate that tells a "
                         "genuine 4000-step student from a snapshot or a substitute with copied metadata "
                         "(same card class; a different class gives a different model, REPRODUCE.md)")
    args = ap.parse_args()

    bad: list = []
    train_argv, head_argv, seeds, head_mode, teacher_seeds, student_argv = resolved_recipe()
    print(f"resolved: head method '{head_mode}'" + (f", teachers {teacher_seeds}" if head_mode == "student" else ""))
    if head_mode == "student" and len(set(teacher_seeds)) != len(teacher_seeds):
        bad.append("duplicate teacher seeds")
        print("  *** TEACHER_SEEDS repeats a seed")
    for var, argv in (("TRAIN", train_argv), ("HEAD", head_argv), ("STUDENT", student_argv)):
        lit = literal_flags(var)
        if argv != lit:
            extra = [t for t in argv if t not in lit]
            print(f"note: {var} resolves with overrides in effect ({' '.join(extra)}); the "
                  f"literal alone is {' '.join(lit)}")
    print(f"resolved: {len(seeds)} member seed(s) {seeds}")
    if len(set(seeds)) != len(seeds):
        bad.append("duplicate seeds")
        print("  *** MEMBERS repeats a seed; identical members are not an ensemble")

    members: list[tuple[str, str | None]]
    head: tuple[str, str | None]
    if args.member or args.head:
        members = [(m, None) for m in args.member]
        head = (args.head, None) if args.head else ("", None)
        print("Checking the explicitly named checkpoints, not the pack's provenance.")
    else:
        pk = torch.load(args.pack, map_location="cpu", weights_only=False)
        if "member_paths" not in pk or "head_path" not in pk:
            raise SystemExit(f"{args.pack} records no member_paths/head_path (packed before "
                             f"2026-09-01); name checkpoints with --member/--head instead")
        shas = pk.get("member_sha256") or []
        if (len(shas) != len(pk["member_paths"]) or not pk.get("head_sha256")
                or not all(SHA_RE.match(str(h)) for h in list(shas) + [pk.get("head_sha256")])):
            raise SystemExit(f"{args.pack} does not record a well-formed SHA256 for every member and "
                             f"the head; a pack without hashes cannot be tied to its checkpoints")
        if len(set(os.path.realpath(p) for p in pk["member_paths"])) != len(pk["member_paths"]):
            raise SystemExit(f"{args.pack} packs the same member checkpoint more than once")
        members = list(zip(pk["member_paths"], shas))
        head = (pk["head_path"], pk["head_sha256"])
        print(f"pack: {args.pack}\n  {len(members)} member(s), head {head[0]}")
        if len(members) != len(seeds):
            bad.append("member count")
            print(f"  *** the pack holds {len(members)} members, the script builds {len(seeds)}")
        hm = [os.path.realpath(p) for p in pk.get("head_member_paths") or []]
        if hm and hm != [os.path.realpath(p) for p, _ in members]:
            bad.append("head members")
            print("  *** the head was calibrated on different members than the pack holds")

    tp, hp = train_mixed.build_parser(), train_final_head.build_parser()
    member_ckpts, head_ckpt = [], None
    member_shas = []
    for i, (path, sha) in enumerate(members):
        ck = load_ckpt(f"member {i}", path, sha, bad)
        if ck is not None:
            member_ckpts.append(ck)
            member_shas.append(sha if sha else sha256(os.path.realpath(path)))
            bad += compare(f"forecaster member {i} (seed {ck['args'].get('seed')})", tp, train_argv, ck["args"])
    if len({ck["args"].get("seed") for ck in member_ckpts}) != len(member_ckpts):
        bad.append("member seeds")
        print("  *** two packed members record the same seed")
    actual_seeds = sorted(str(ck["args"].get("seed")) for ck in member_ckpts)
    if actual_seeds != sorted(seeds):
        bad.append("member seed contract")
        print(f"  *** member seeds {actual_seeds} differ from contract {sorted(seeds)}")
    if head[0]:
        ck = load_ckpt("interval head", head[0], head[1], bad)
        if ck is not None:
            head_ckpt = ck
            is_student = bool(ck.get("teachers"))
            if head_mode == "student" and not is_student:
                bad.append("head method")
                print("  *** the pack ships a plain fitted head, but the script builds a distilled student")
            if head_mode == "single" and is_student:
                bad.append("head method")
                print("  *** the pack ships a distilled student, but the script builds a single fitted head")
            if is_student:
                bad += check_student(ck, student_argv, head_argv, teacher_seeds, member_shas, hp, bad_list=bad)
                hv = {}
            else:
                bad += compare("interval head", hp, head_argv, ck["args"])
                hv, _, _ = parsed(hp, head_argv)
            if is_student:
                pass
            elif "step" not in ck:
                # Heads saved before 2026-09-11 record no step; their age rests on the
                # recorded fixed_step argument, which `compare` has already checked.
                print("  note: this head records no checkpoint step (saved before 2026-09-11)")
            elif hv.get("fixed_step") and ck["step"] != hv["fixed_step"]:
                bad.append("head step")
                print(f"  *** the head checkpoint is step {ck['step']}, the script keeps step "
                      f"{hv['fixed_step']}")
            hpaths = [os.path.realpath(p) for p in ck.get("paths") or []]
            if hpaths and members and hpaths != [os.path.realpath(p) for p, _ in members]:
                bad.append("head paths")
                print("  *** the head checkpoint records different member paths than the pack")
            # Hashes recorded when the head was FITTED (train_final_head, 2026-09-11) against
            # the members' hashes at packing: a member replaced between the two passes a
            # path comparison and fails here. Older heads record none; say so.
            fit = ck.get("member_sha256")
            if fit is None:
                print("  note: this head predates fit-time member hashes; provenance rests on paths")
            elif members and list(fit) != member_shas:
                bad.append("head fit hashes")
                print("  *** the members the head was fitted on are not the packed members (hash)")
            else:
                print("  head fit-time member hashes match the packed members")
    if args.replay_student:
        if head_ckpt is None or not head_ckpt.get("teachers"):
            bad.append("replay"); print("\n=== student replay: no student checkpoint to replay")
        else:
            bad += replay_student(head[0], head_ckpt)
    if not (args.member or args.head):
        inference = args.inference or os.path.join(os.path.dirname(args.pack), "submission.py")
        want = champion_source()
        print(f"\n=== inference source: {inference}\n  contract (CHAMPION): {os.path.normpath(want)}")
        if not (os.path.isfile(inference) and os.path.isfile(want) and sha256(inference) == sha256(want)):
            bad.append("inference source identity")
            print(f"  *** {inference} is not the contract's inference source byte for byte")
        else:
            print(f"  identical, sha256 {sha256(want)[:16]}")
        verify_kernel_images(inference, bad)
        verify_pack_schema(pk, bad)
        if len(member_ckpts) == len(members) and head_ckpt is not None:
            verify_pack_tensors(pk, member_ckpts, head_ckpt, bad)
        else:
            bad.append("tensor verification skipped")
            print("\n=== packed tensors: not verified because a checkpoint failed to load")
    data_provenance(bad)

    print("\n" + ("OK -- train_final.sh as it resolves now matches every packed member, the head "
                  "and the release data." if not bad else
                  f"FAILED -- {len(bad)} disagreement(s). Either the script is stale "
                  f"or the shipped build was not made with it."))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
