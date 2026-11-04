"""Which recorded runs used a teacher or a warm start, and did any of it leak?

Distillation and warm starts both import whatever the source checkpoint saw, and
until 2026-09-01 neither was guarded on more than two holdout axes. This walks every
`best.pt` under the runs tree and pairs each student/warm-started run with its source.

**Result of the 2026-09-01 sweep over 2580 checkpoints: no real leak.** 100
distillation students and 3 warm starts, and every flag was a false positive:

  * `kf_dist_only_rp*` teachers predate the holdout-env record, so they report
    `aoa=None re=None` -- but their run directories name the same Re fold as the
    student (`kf_cap_w128_12k_rp1_3750_16500_s0` for the `rp1` student), so they
    were per-fold teachers all along.
  * `seq_s0..s2` warm-start from `simpre_s*`, which ran `--mode sim_only`. That
    empties `real_trajs` whatever the flag says, so the recorded `real_trajs="all"`
    describes a run that saw no real data at all.

Both classes are now handled in `train_mixed.py`'s guards rather than reported here:
unverifiable provenance warns instead of failing, and a `sim_only` source counts as
having seen nothing. Re-run this after any distillation campaign.
"""
import glob, os, torch
R = "runs"
tea, warm, n = [], [], 0
for p in sorted(glob.glob(f"{R}/*/best.pt")):
    try:
        ck = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
        a = ck["args"]
    except Exception:
        continue
    n += 1
    d = os.path.basename(os.path.dirname(p))
    if a.get("distill_from"):
        tea.append((d, str(a["distill_from"]), a.get("real_trajs"),
                    a.get("REALISTIX_HELDOUT_AOA"), a.get("REALISTIX_HELDOUT_RE")))
    if a.get("init_from"):
        warm.append((d, str(a["init_from"]), a.get("real_trajs"),
                     a.get("REALISTIX_HELDOUT_AOA"), a.get("REALISTIX_HELDOUT_RE")))
print(f"scanned {n} run checkpoints")

def src_args(path):
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)["args"]
    except Exception:
        return None

print(f"\n=== {len(tea)} distillation students ===")
for d, t, rt, ao, re_ in tea:
    sa = src_args(t)
    s = (f"teacher real_trajs={sa.get('real_trajs')} aoa={sa.get('REALISTIX_HELDOUT_AOA')!r} "
         f"re={sa.get('REALISTIX_HELDOUT_RE')!r}") if sa else "teacher UNREADABLE"
    leak = ""
    if sa:
        if sa.get("real_trajs") == "all" and rt != "all":
            leak = "  <-- LEAK: teacher saw all data"
        elif (sa.get("REALISTIX_HELDOUT_AOA") or "") != (ao or "") or \
             (sa.get("REALISTIX_HELDOUT_RE") or "") != (re_ or ""):
            leak = "  <-- LEAK: teacher holdout differs"
    print(f"  {d:<30} student rt={rt} aoa={ao!r} re={re_!r} | {s}{leak}")

print(f"\n=== {len(warm)} warm-started runs ===")
for d, t, rt, ao, re_ in warm:
    sa = src_args(t)
    s = (f"source real_trajs={sa.get('real_trajs')} aoa={sa.get('REALISTIX_HELDOUT_AOA')!r} "
         f"re={sa.get('REALISTIX_HELDOUT_RE')!r}") if sa else "source UNREADABLE"
    leak = ""
    if sa:
        if sa.get("real_trajs") == "all" and rt != "all":
            leak = "  <-- LEAK: source saw all data"
        elif (sa.get("REALISTIX_HELDOUT_AOA") or "") != (ao or "") or \
             (sa.get("REALISTIX_HELDOUT_RE") or "") != (re_ or ""):
            leak = "  <-- LEAK: source holdout differs"
    print(f"  {d:<30} run rt={rt} aoa={ao!r} re={re_!r} | {s}{leak}")
