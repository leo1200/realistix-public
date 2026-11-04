#!/usr/bin/env python3
"""Predict a candidate's leaderboard score, and keep the predictor honest over time.

**Status: the subscore predictor is UNCALIBRATED.** It fails both back-tests it can
currently be checked against (`backtest`). Use it to *screen* -- it is the only
instrument that gives a candidate a split it did not train on -- and not to promote.
Every submission adds a row, and `fit` re-estimates the transfer; three of four
back-tests agreeing is the bar for trusting a magnitude.

What is solid, and what is not:

* **Solid.** `final_score` is a linear combination of the five subscores, fitted on
  every scored submission, leave-one-out RMS ~0.011. Given subscores, the total
  follows. Also solid: the board's noise floor, ~0.05 per subscore, measured by
  892587 re-training 887158's recipe from scratch and scoring within
  +0.010/−0.022/+0.047/−0.041.
* **Not solid.** Predicting the *subscores* of a change we have not shipped. Two
  instruments have failed: the regime folds (wrong sign on distillation) and the
  honest tail pipeline (wrong sign on distillation, 6.5x too large on
  `--mults 1,2`).

Why it is hard, stated once so nobody re-derives it: since 882969 every shipped
model trains on every usable trajectory and every frame, so no local split is
out-of-sample for it. `tools/honest_pipeline.sh` builds the *same recipe* under a
withheld tail so a candidate can be measured honestly, and this module scores both
that artefact and the all-data one -- the gap between them is the overfitting
signal, and distillation is the case that showed why it matters (+0.18 rel_l2
in-sample against +0.016 honest).

    python tools/board_model.py score submissions/combiner --tag combiner
    python tools/board_model.py predict --tag combiner --ref all82_spread
    python tools/board_model.py record --tag combiner --id 895xxx \
        --board 94.83,78.03,94.00,90.38,42.36
    python tools/board_model.py backtest
    python tools/board_model.py fit
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("REALISTIX_RUNS", "runs")
REGISTRY = os.path.join(RUNS, "board_registry.json")
EVAL = "eval"
PY = os.environ.get("PY", "python")

SUB = ("rel_l2", "tke", "mvpe", "time", "sps")
ACC = ("rel_l2", "tke", "mvpe", "sps")

# Seeded from docs/archive/HANDOFF_2026-09-10.md section 2. `local` is filled in by `score`.
SEED = {
    "unet":         dict(board_id=879743, board=[93.914, 77.745, 93.078, 91.551, 33.265]),
    "ensemble":     dict(board_id=881057, board=[94.300, 77.460, 93.310, 90.000, 39.213]),
    "ensemble96":   dict(board_id=882102, board=[94.274, 78.006, 93.544, 89.978, 39.489]),
    "all82_spread": dict(board_id=887158, board=[94.834, 78.030, 93.999, 90.379, 42.362]),
    "repro818":     dict(board_id=892587, board=[94.844, 78.008, 94.046, 90.266, 42.321]),
    "onemem":       dict(board_id=893273, board=[94.752, 77.958, 93.820, 91.331, 41.911]),
    "distilled":    dict(board_id=894861, board=[94.717, 77.844, 93.238, 90.248, 42.366]),
}


def load():
    if os.path.exists(REGISTRY):
        return json.load(open(REGISTRY))
    return {k: dict(v) for k, v in SEED.items()}


def save(reg):
    json.dump(reg, open(REGISTRY, "w"), indent=1, sort_keys=True)


def weights(reg):
    """final = w . subscores, refit on every row that has a board result."""
    rows = [r for r in reg.values() if r.get("board") and r.get("final")]
    if len(rows) < 6:
        return None, None
    A = np.array([r["board"] for r in rows]); y = np.array([r["final"] for r in rows])
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    loo = []
    for i in range(len(rows)):
        m = np.ones(len(rows), bool); m[i] = False
        wi, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
        loo.append(A[i] @ wi - y[i])
    return w, float(np.sqrt(np.mean(np.square(loo))))


def run_score(sub_dir, eval_dir):
    """Score a packed submission the way the leaderboard does."""
    out = subprocess.run([PY, os.path.join(HERE, "run_and_score.py"),
                          os.path.join(sub_dir, "submission.py"), eval_dir],
                         capture_output=True, text=True, cwd=os.path.join(HERE, ".."))
    got = {}
    for line in (out.stdout or "").splitlines():
        p = line.split()
        if len(p) == 2 and p[0].endswith("_score"):
            got[p[0][:-6]] = float(p[1])
        if len(p) == 3 and p[1] == "ms/sample":
            got["ms"] = float(p[0])
    if not got:
        raise SystemExit(f"run_and_score produced nothing for {sub_dir}\n{out.stdout}\n{out.stderr}")
    return got


def cmd_score(a):
    reg = load()
    row = reg.setdefault(a.tag, {})
    for name, d in (("in_sample", "val_aoa"), ("honest", "ingrid")):
        p = os.path.join(EVAL, d)
        if not os.path.exists(p):
            print(f"  {name}: no eval set at {p}, skipped"); continue
        row[name] = run_score(a.sub, p)
        print(f"  {name:>10} ({d}): " + "  ".join(
            f"{k} {row[name][k]:.3f}" for k in ACC if k in row[name]))
    save(reg)
    print(f"\nwrote {a.tag} to {REGISTRY}")


def cmd_predict(a):
    reg = load()
    if a.tag not in reg or "honest" not in reg[a.tag]:
        raise SystemExit(f"score {a.tag} first")
    if a.ref not in reg or "honest" not in reg[a.ref]:
        raise SystemExit(f"score the reference {a.ref} first")
    cand, ref = reg[a.tag], reg[a.ref]
    w, loo = weights(reg)
    cal = reg.get("_calibration", {})

    print(f"predicting '{a.tag}' against '{a.ref}'\n")
    print(f"{'subscore':>9}{'honest d':>11}{'in-sample d':>13}{'overfit':>9}"
          f"{'x transfer':>12}{'board d':>10}")
    tot, warn = 0.0, []
    for i, k in enumerate(ACC):
        dh = cand["honest"][k] - ref["honest"][k]
        di = (cand["in_sample"][k] - ref["in_sample"][k]
              if "in_sample" in cand and "in_sample" in ref else float("nan"))
        t = cal.get(k, 1.0)
        db = t * dh
        if np.isfinite(di) and abs(di) > 3 * max(abs(dh), 1e-6):
            warn.append(k)
        if w is not None:
            tot += w[SUB.index(k)] * db
        print(f"{k:>9}{dh:+11.3f}{di:+13.3f}{'!' if k in warn else '':>9}"
              f"{t:12.2f}{db:+10.3f}")
    if w is not None and "ms" in cand.get("honest", {}) and "ms" in ref.get("honest", {}):
        import math
        def ts(t_ms): return 100 / (1 + math.sqrt(t_ms / 1000 / 0.72896))
        # Board time is not local time; scale the *delta* by the measured
        # local-to-container factor of 1.71 (submission 893273) around the board's
        # current operating point.
        d_ms = (cand["honest"]["ms"] - ref["honest"]["ms"]) * 1.71
        d_t = ts(8.261 + d_ms) - ts(8.261)
        tot += w[SUB.index("time")] * d_t
        print(f"{'time':>9}{d_ms:+11.3f}{'':>13}{'':>9}{'ms->score':>12}{d_t:+10.3f}")
    print(f"\n  predicted board delta {tot:+.3f}"
          f"   (weights LOO {loo:.4f})" if w is not None else "")
    if warn:
        print(f"  !! in-sample gain exceeds honest gain by >3x on {', '.join(warn)}"
              f" -- that is the distillation signature; treat the prediction as an "
              f"upper bound")
    print("  !! the per-subscore transfer is UNCALIBRATED (see backtest); this is a "
          "screen, not a projection")


def cmd_record(a):
    reg = load()
    row = reg.setdefault(a.tag, {})
    row["board_id"] = a.id
    row["board"] = [float(v) for v in a.board.split(",")]
    if a.final is not None:
        row["final"] = a.final
    save(reg)
    print(f"recorded {a.tag} -> board {row['board']}")


def cmd_backtest(a):
    reg = load()
    rows = [(t, r) for t, r in reg.items()
            if isinstance(r, dict) and r.get("board") and r.get("honest")]
    print(f"{len(rows)} rows with both a board result and an honest local score\n")
    if len(rows) < 2:
        print("  not enough to back-test. Score the archived artefacts with the"
              "\n  honest pipeline first (tools/honest_pipeline.sh), then re-run.")
        return
    print(f"{'tag':>14}{'subscore':>10}{'honest d':>11}{'board d':>10}{'ratio':>9}")
    ref = dict(rows).get("all82_spread")
    for t, r in rows:
        if t == "all82_spread" or not ref:
            continue
        for k in ACC:
            dh = r["honest"][k] - ref["honest"][k]
            db = r["board"][SUB.index(k)] - ref["board"][SUB.index(k)]
            ratio = db / dh if abs(dh) > 1e-6 else float("nan")
            print(f"{t:>14}{k:>10}{dh:+11.3f}{db:+10.3f}{ratio:9.2f}")


def cmd_fit(a):
    reg = load()
    rows = [(t, r) for t, r in reg.items()
            if isinstance(r, dict) and r.get("board") and r.get("honest")]
    ref = dict(rows).get("all82_spread")
    if not ref or len(rows) < 4:
        print("need at least 4 artefacts scored honestly, including all82_spread")
        return
    cal = {}
    for k in ACC:
        x, y = [], []
        for t, r in rows:
            if t == "all82_spread":
                continue
            x.append(r["honest"][k] - ref["honest"][k])
            y.append(r["board"][SUB.index(k)] - ref["board"][SUB.index(k)])
        x, y = np.array(x), np.array(y)
        if np.abs(x).max() < 1e-6:
            continue
        cal[k] = float((x @ y) / (x @ x))          # slope through the origin
        print(f"  {k:>7}: transfer {cal[k]:+.3f}  (n={len(x)})")
    reg["_calibration"] = cal
    save(reg)
    print(f"\nwrote calibration to {REGISTRY}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("score"); s.add_argument("sub"); s.add_argument("--tag", required=True)
    s.set_defaults(fn=cmd_score)
    s = sp.add_parser("predict"); s.add_argument("--tag", required=True)
    s.add_argument("--ref", default="all82_spread"); s.set_defaults(fn=cmd_predict)
    s = sp.add_parser("record"); s.add_argument("--tag", required=True)
    s.add_argument("--id", type=int, required=True); s.add_argument("--board", required=True)
    s.add_argument("--final", type=float); s.set_defaults(fn=cmd_record)
    s = sp.add_parser("backtest"); s.set_defaults(fn=cmd_backtest)
    s = sp.add_parser("fit"); s.set_defaults(fn=cmd_fit)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
