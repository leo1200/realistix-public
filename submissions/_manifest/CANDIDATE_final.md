# Candidate `final` — Decision-Phase inference interface on the incumbent weights (built 2026-09-11, NOT uploaded)

**What it is.** `submissions/final/submission.py` = wd03mv's inference code plus the interface repairs the audits
asked for, packed with a byte copy of 918211's `model.pth` (`build/final.zip`, 149.8 MB, 162.1 MB extracted).
The learned function is unchanged: outputs are **bit-identical** to wd03mv's at N = 1, 2, 3, 4, 8, 9, 16, 17, 64, 65
(`research/audit_2026-09-11/claude/smoke_final.json`); `check_pack` MATCH (mean relative 2.95e-4 vs fp32 members).

**Changes.** (1) `SubmissionModel(device=)` and `load_checkpoint(path, device)` are honoured: a CUDA device string
selects that device (an out-of-range index raises `ValueError` instead of silently reloading elsewhere); each
instance predicts on its own device; "cpu" is honoured when CUDA is absent. **Disputed (Codex round 2):** "cpu" on
a CUDA host is read as "no preference" unless `REALPDE_FORCE_CPU=1`, because the documented signature *defaults*
`device` to "cpu" and an unset argument is indistinguishable from an explicit one; honouring it literally could
move a GPU-provisioned evaluation to the CPU. Codex (round 3) accepts this as a provisional, reversible compatibility decision. The organisers'
evaluation example (~09-22) settles which argument the ingestion actually passes; the decision is one line.
(2) One 60 s deadline covers load, warm-up, every graph capture and the self-check (wd03mv budgeted capture
alone), checked before each phase; past it, uncaptured or unchecked graphs are discarded and predict() runs the
eager path. (3) One re-entrant lock serialises setup, prediction and teardown (concurrent replay into the shared
static buffers was unsafe in wd03mv too; 3 threads × 30 mixed-size calls now reproduce the serial outputs exactly).
(4) `_load(None)` never re-resolves the default device over an explicit choice. Probes:
`research/audit_2026-09-11/claude/{smoke_final,device_race_final}.json`; Codex rounds 2–3 under `../codex/`.
Standalone extraction of the zip in a fresh process: no repo imports, outputs identical to wd03mv, `check_pack`
(prediction + intervals) MATCH.

**Pre-registered prediction (if it takes a slot).** rel / tke / mvpe / sps identical to 918211's board subscores
(94.847 / 78.913 / 94.092 / 43.723) to the last digit; time_score within the same-artefact noise band of
918211's 92.174 (±0.04 of final, i.e. 5.0–5.6 ms/sample). **Kill:** any accuracy subscore differing from 918211's,
or time_score below 91.4 (the P1 read, 6.5 ms) — either means the harness exercised a path the smoke did not.
Information value: confirms the class/device interface and the setup deadline behave under the real ingestion;
it cannot improve the score. Codex round-2 review (`research/audit_2026-09-11/codex/codex_report2.md`) precedes
any upload; the owner decides the slot.

## Rebuilt on the four-member pack (2026-09-12 02:55 UTC)
`submissions/final/model.pth` is now a byte copy of `submissions/k4/model.pth` (923184's weights); `build/final.zip`
191.6 MB / 207.4 MB extracted. Smoke against `submissions/k4`: outputs bit-identical at N = 1…65, repeated/class calls
exact, eager == graph, expired deadline → eager, CPU-only fp32; `check_pack` (prediction + intervals) MATCH.
**Prediction if it takes a slot:** rel / tke / mvpe / sps identical to 923184's (94.853 / 78.977 / 94.101 / 43.841) to the
last digit; time within ±0.04 of final of 92.164. Kill: any accuracy digit differs, or time_score < 91.4.
