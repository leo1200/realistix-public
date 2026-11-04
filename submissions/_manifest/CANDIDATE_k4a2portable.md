# Candidate k4a2portable — 929146's weights with the preprocessing computed once and shared (pre-registered 2026-09-17, Codex round 3 + Claude; CONDITIONAL, not armed)

**What.** `research/audit_2026-09-17/codex/round3/k4a2portable.zip` (copy: `build/k4a2portable.zip`),
sha256 `d7c00cb5b58b51c219efa6753e676add992c240785901349a4859abe3b8609cd`, 191,640,308 bytes; `model.pth` **byte-identical
to `submissions/k4a2/model.pth`** (`ce6b41c8…1f1f8`). Same four members, same distilled head, gain dose 0.5 / cap 3, same
fp32 statistics, interval construction, containment guard and three output copies. The only change: the per-window
normalisation (`_pre`) is computed once in its own captured graph and its outputs shared by the four member graphs and
the head, instead of being recomputed inside each member graph (4 × 17 redundant nodes). **Torch-only** — no Triton,
no TorchScript, no compiler dependence (`PREREG_PORTABLE_A2.md` in the same directory has the full gate list).

**Why this and not the Triton-fused gain.** Round 2's Triton kernel removed the whole local contention penalty, but
round 3 showed the shipped image (`pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`, conda torchtriton 2.2.0, **no C
compiler**) makes it unsafe: Triton's cache key includes the `libtriton.so` hash, the pip build's launchers do not match
the image's (cugraph16t: 4/4 archived keys match pip, 0/4 the image), so an empty or mismatched cache means an attempted
host compile → no compiler → fallback. That is the concrete route by which cugraph16t fell back to the unfused path; a
guarded Triton variant exists but is not an approved substitute. The torch-only shared-`_pre` change, measured on a pq
A100-PCIE-40GB (sm_80, balanced same-process protocol, means; `round3/a100_final_fourway.json`):

| N=1 quiet | k4 5.0996 | k4g 5.1552 | **portable 5.0145** | Triton 5.0841 |
| N=1 contended | k4 6.9891 | k4g 7.0714 | **portable 6.9714** | Triton 7.0411 |

It removes more than the whole k4g-minus-k4 penalty and beats k4 itself (−0.085 / −0.018 ms). H100 reads are less
stable under contention (42–100 % of the penalty removed across runs). A100 deployment gates passed: nine graphs,
zero eager fallbacks, read-only artefact, no Triton import, import 14.2 s, prediction/lower/upper within 2.44e-4 of A2 on
all 1,672 release windows, release subscores identical to four decimals (deltas ≤ 4.5e-6).

**Prediction (exact power mean).** Accuracy = **929146's digits** (rel 94.8293790855 / sps 43.9453064676 /
tke 79.1320014641 / mvpe 94.1013126942); an accuracy move > 1e-5 on any subscore **quarantines** the read (falsifies
"time only"). Clock: **if the 09-18 control shows the gain chain pays** → 5.34 ms [5.10, 5.60] → final **81.831**
[81.814, 81.847]; **if shared drift** → 5.83 ms [5.40, 6.25] → 81.800 [81.774, 81.827]. Beats the retained best
(926634, 81.80680) iff the clock draws below **5.7168 ms**.

**Condition and decision rule.** Upload on 09-19 **only if** the 09-18 byte-identical 923184 repeat reads **≤ 5.40 ms**
(`CANDIDATE_k4repeat.md`). A control ≥ 5.60 (shared drift) or in between (inconclusive) does not open this slot; then
09-19 is a registered exact A2 repeat or a retrain calibration. Kill: accuracy outside 1e-5, execution failure, or a
clock outside both hypothesis bands. Time ≥ 5.7168 ms is an economic failure to beat the retained best even inside the
drift band. One read does not identify a platform regime; any confirmation must reuse the exact bytes.

## Armed
(not armed — waits for the 09-18 control; `research/audit_2026-09-17/claude/arm_k4a2portable.sh` is ready)

**Archive identity (2026-09-17, after Codex round 4).** Round 3 also produced `k4a2portable_pinned.zip`
(sha256 `d80f2dd1…7843e`: shared preprocessing *plus* pinned non-blocking output copies). Round 4's slot note names
it; round 3's own selection did not — the pinned combination was less robust under CPU contention (73 % of the fused
saving vs 245 % quiet) and changes the host-buffer semantics. **The registered artefact is `k4a2portable.zip`,
sha256 `d7c00cb5…609cd` (`candidate_receipt.json`: variant `pre_once`, original output path), and the arming script
is bound to that hash.** Any other variant needs its own manifest before it may use a slot.
