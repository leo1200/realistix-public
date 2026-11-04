# Candidate dec12klog — the Decision-Phase deliverable itself, with stderr diagnostics (registered 2026-09-24 05:40 UTC)

**What it is.** `submissions/dec12k/model.pth` byte for byte (the `train_final.sh` rebuild from commit a49588c into
an empty artefact root: four members bitwise identical to canon0918's 544/544, six teachers bitwise identical to
canon0918's, the 12000-step student) with dec12k's `submission.py` plus **stderr-only logging**: import duration,
device name, and for the first four calls per process the input shape, the path (graph/eager) and the internal
predict time. The FAQ (changed after 2026-09-22) explicitly allows logging "the device, input shapes and timings".
Outputs verified **bitwise identical** to dec12k on N = 1 / 8 / 37 in fresh processes
(`research/final_2026-09-24/parity_zip.sh`, 9/9 arrays, float32).

**Why this slot.** (1) The board artefact should be the deliverable the organisers will rebuild, not 934147's
pre-commit bytes, which no longer replay. (2) The FAQ now says every scored batch runs in a *fresh process* that
re-imports `submission.py` (≈ 16 s here with graph capture). How many processes and calls a run makes on the
evaluation host (an A800-SXM4-80GB, per 940198's ingestion log) prices the Decision-Phase wall-clock risk on a
larger private set; the logs answer it, and nothing else can.

**Prediction.** rel **94.8462054793** and mvpe **94.0909748382** exactly (members, gain and fp32 return path
identical to 934147/940198); tke 79.23286 ± 2e-5; sps **44.00 ± 0.01** (a different but equivalent 12k student to
940198's; fold/in-sample equivalence measured at three decimals); clock a draw from N(4.85, 0.25) ms, plus ≤ 0.01 ms
for the log writes. Final ≈ 81.90 ± 0.02. **Kill/quarantine:** rel or mvpe differing from the digits above means the
pack is not what was checked — stop and investigate before any further upload.

## Read — 942498, uploaded 05:40 UTC, scored ~05:50: **81.90992, new retained best**
| | rel | tke | mvpe | sps | time (ms) | final |
|---|---|---|---|---|---|---|
| 934147 (previous best) | 94.8462054793 | 79.2328683384 | 94.0909748382 | 43.9750022266 | 4.649 | 81.90253 |
| 940198 (H100 12k student) | 94.8462054793 | 79.2328645965 | 94.0909748382 | 44.0012119293 | 4.754 | 81.90147 |
| **942498 (dec12klog)** | **94.8462054793** | 79.2328627255 | **94.0909748382** | **44.0016155224** | **4.636** | **81.90992** |

Every registered digit held: rel and MVPE exact, TKE within 6e-6, SPS 44.0016 inside 44.00 ± 0.01 (a different but
equivalent 12k student lands 0.0004 from 940198's). The clock drew 4.636 ms. `check_recipe --replay-student` on this
pack: **80/80 bitwise** on node7 (built on node10 a day earlier) — the integrity chain holds for the board artefact.

**What the logs show** (`research/final_2026-09-24/logs_942498/`): the host is an A800-SXM4-80GB; the dev run is
**17 fresh processes, each one warm-up + one timed call of N = 5** (85 windows); import **3.9 s** there (so even 10× the
batches is ~11 min of imports -- note the 5-min dev wall would not survive 10x, but the dev set is far inside it);
our predict **8.4–8.9 ms per timed call = 1.7 ms/sample**, against the harness's reported 4.6 ms/sample. So about
**37 % of the timed region is our own code** under the current harness, and N = 5 is served by the 6-bucket (one
padded window).
