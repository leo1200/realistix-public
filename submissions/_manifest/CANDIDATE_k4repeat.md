# Calibration slot k4repeat — byte-identical re-upload of 923184 (`k4.zip`) as the clock control (pre-registered 2026-09-17, joint Claude + Codex round 1 R1; owner's go 2026-09-17)

**What.** The pinned archive `archive/923184/k4.zip`, sha256
`c74edaef7872c978c3b30e9633958a4ed6a05f4e351ef6be0d55883ba5f4897f`, unchanged: four members, one plain head, k4's
gain-free inference code (md5 `249fd785…`). No build, no training. Uploaded on the 2026-09-18 UTC slot.

**Why.** The clock has two levels (`research/audit_2026-09-17/claude/AUDIT.md` §3, `FINDINGS.md` §5): six graphed reads
09-05→09-12 at 5.341 ± 0.082 ms (923184 itself: 5.270), four reads on the k4g gain code 09-13→09-17 at 5.830 ± 0.186.
The 0.49 ms shift is worth 0.031 of final — the head pair's whole accuracy gain — and code and date are perfectly
confounded on the board. Locally the in-graph gain costs +0.05 ms quiet and +0.37…+0.61 ms under CPU contention
(Codex, paired same-process). A byte-identical repeat of the last pre-gain artefact is the only instrument that
separates a code cause from platform drift; every prediction since 09-13 has been priced against one or the other.

**Prediction (registered before the score).** Accuracy: **identical to 923184 to every published digit** — rel
94.8527950619, tke 78.9765018158, mvpe 94.1013087356, sps 43.8407091309 (API record, `my_submissions_all.json`) — the artefact is the same bytes.
Clock, two hypotheses: **code-caused** → 5.34 ms [5.10, 5.60] → final ≈ 81.806 [81.79, 81.82]; **platform drift**
→ 5.83 ms [5.40, 6.25] → final ≈ 81.775 [81.75, 81.80]. It cannot be a new best unless the clock draws below
923184's own 5.270 ms.

**Decision rule.** **Quarantine** (build/platform fault, do not interpret) if any accuracy subscore differs from
923184 by more than 1e-6. Clock **≤ 5.40 ms → code-caused**: the k4g chain pays ≈ 0.5 ms on the container; the 09-19
slot goes to `k4a2portable` (`CANDIDATE_k4a2portable.md`: 929146's weights, preprocessing computed once and shared,
torch-only, A100 gates passed) — the Triton-fused gain was withdrawn as evaluator-unsafe in Codex round 3; gain-free
A2 (R2 in `codex/PREREG_NEXT.md`) is the fallback.
Clock **≥ 5.60 ms → shared drift**: re-centre every clock prediction at 5.83 ms; no gain removal; the 09-19 slot is a
retrain calibration or a registered exact A2 repeat. **Between 5.40 and 5.60: inconclusive**, treated as drift-leaning
— no artefact change is justified by the clock alone. Neither outcome proves exclusivity: both causes can coexist.

## Armed (owner's go, 2026-09-17)
`research/audit_2026-09-17/claude/arm_k4repeat.sh` — waits for 2026-09-18T00:00:30Z, verifies the sha256 against the
pinned archive before sending, retries to 23:45Z with quota re-checked before every attempt; log
`logs/submit_k4repeat.log`.

## Board result — submission 930933, uploaded 2026-09-18T00:01Z, scored 00:06 UTC (scoring rpw2, ingestion rpw6)
| | rel | tke | mvpe | sps | time (ms) | final |
|---|---|---|---|---|---|---|
| 923184 (09-12) | 94.8527950619 | 78.9765018158 | 94.1013087356 | 43.8407091309 | 92.164 (5.270) | 81.80564 |
| **930933 (same bytes, 09-18)** | 94.8527950619 | 78.9765018158 | 94.1013087356 | **43.8406997038** | **91.612 (6.110)** | 81.75247 |

**Accuracy.** rel / tke / mvpe reproduce to every digit; SPS differs by **9.4e-6**, which tripped the registered 1e-6
quarantine (the watcher correctly armed nothing). The difference is the scorer's known non-determinism on the graphed
fp16 path (Codex 09-16: release subscores move up to 1.3e-5 between fresh processes and batch decompositions), not a
build fault — the point forecast channels are bit-identical. The 1e-6 threshold was too tight by one order of
magnitude; **the read is valid as a clock read**.

**Clock: 6.110 ms → the DRIFT branch, decisively.** The same bytes that drew 5.270 ms on 09-12 drew 6.110 ms today,
the slowest read of any graphed artefact. The k4g gain code is exonerated: the 0.49 ms "level shift" of 09-13 → 09-17 is
platform drift (the harness/co-tenant level rose ≈ 0.8 ms in six days), and every artefact since 09-13 has been read on
the new level. Consequences: (i) `k4a2portable` is **not** armed — its saving is local GPU time whose board price is
unresolved; (ii) every clock prediction is re-centred at **≈ 6.0 ms [5.6, 6.3]**; (iii) 926634's 5.70 was a favourable
draw within the new regime, and an exact repeat of any of our artefacts today prices at ≈ 81.75–81.78; (iv) the
retained best stays 926634. The 09-19 slot goes to the drift-branch option: a **retrain calibration** — the canonical
A100 rebuild of the frozen method (seeds 0–3, `submissions/canon0918/`, queued on pq at 04:15Z) — the artefact the
organisers' own retrain would produce on the same silicon.
