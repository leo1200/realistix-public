# Research rounds

A curated record: one report per round of the competition's August–October 2026 development, and the post-competition
simulation study in full. The rounds were run as joint passes of the author with Claude Code and OpenAI Codex; the
reports cite briefs, scripts, run directories and per-run reads that are not part of this release, so treat paths in
them as references, not links. The consolidated results are in [`../FINDINGS.md`](../FINDINGS.md) and
[`../docs/REPORT.md`](../docs/REPORT.md); the August–September record is in [`../docs/archive/`](../docs/archive/).

| round | report | what it tested | outcome |
|---|---|---|---|
| 2026-09-05 potential | [`potential_2026-09-05/claude/claude_analysis.md`](potential_2026-09-05/claude/claude_analysis.md) | where the remaining potential sits; the angle-holdout suite; a broad idea catalogue | angle generalisation opened; stopping step, soups, SAM, meta-updates, stem, morphing screened, all below the bar (`docs/archive/POTENTIAL_2026-09-05.md`) |
| 2026-09-07 optimizer | [`optimizer_2026-09-07/claude/claude_analysis.md`](optimizer_2026-09-07/claude/claude_analysis.md) | nine optimizer arms, physics-loss oracles | Muon weight decay 0.03 shipped; the rest closed |
| 2026-09-09 audit | [`audit_2026-09-09/REPORT.md`](audit_2026-09-09/REPORT.md) | control inventory, cooperative training | cooperative training closed |
| 2026-09-09 repair | [`repair_2026-09-09/REPORT.md`](repair_2026-09-09/REPORT.md) | controls, mean-flow correction | mean-flow correction kept as a calibration slot |
| 2026-09-10 audit | [`../docs/archive/AUDIT_2026-09-10.md`](../docs/archive/AUDIT_2026-09-10.md) | full audit of record, code and conclusions | corrections folded into the living documents |
| 2026-09-10 improve | [`improve_2026-09-10/claude/NOTES.md`](improve_2026-09-10/claude/NOTES.md) | 63 screening runs: sim-SPS split, time dilation, DropPath, frame masking, duration, mean-flow mixing | all closed (`docs/archive/IMPROVE_2026-09-10.md`) |
| 2026-09-15 slot | [`slot_2026-09-15/NOTES.md`](slot_2026-09-15/NOTES.md) | the interval head: its policy, and what to spend a submission slot on | the head's policy is at its optimum; averaging the outputs of two independently fitted heads passes its gate (+0.10 SPS on held-out folds) |
| 2026-09-16 distillation | [`distil_2026-09-16/codex_audit/SUMMARY.md`](distil_2026-09-16/codex_audit/SUMMARY.md) | independent audit of the distilled interval head and its training length | the frozen 4000-step student kept for submission; the 12000-step student better on folds |
| 2026-09-17 audit | [`audit_2026-09-17/codex/REPORT.md`](audit_2026-09-17/codex/REPORT.md) | the scoring formula, the timed inference path | the exact power-mean score recovered; CPU contention on the timed path found |
| 2026-09-18 interval heads | [`leap2_2026-09-18/RESULTS.md`](leap2_2026-09-18/RESULTS.md), [`density_2026-09-18/RESULTS.md`](density_2026-09-18/RESULTS.md) | where the SPS oracle gap is; a mixture-density interval head | the gap sits in the calm 95 % of elements, not in large errors; the density head killed at its gate |
| 2026-09-20 flow donor | [`flow2_2026-09-20/RESULTS.md`](flow2_2026-09-20/RESULTS.md) | a flow-matched forecaster as a fifth, decorrelated member | decorrelated and equally accurate, still below its gate |
| 2026-09-22 push | [`push_2026-09-22/NOTES.md`](push_2026-09-22/NOTES.md) | the plan for the last five days | effort split between Decision-Phase preparation (a joint high-Re / 5° fold, the 12000-step student) and one gated board slot |
| 2026-09-24 last | [`last_2026-09-24/NOTES.md`](last_2026-09-24/NOTES.md) | audit; fused GroupNorm kernel | the fused kernels shipped |
| 2026-09-24 final | [`final_2026-09-24/NOTES.md`](final_2026-09-24/NOTES.md) | deliverable integrity, harness timing | the deliverable reproduced bit for bit across nodes |
| 2026-11-02 post-competition | [`rb_2026-11-02/NOTES.md`](rb_2026-11-02/NOTES.md) | RealPDEBench data; real-coverage dose at a novel angle; the simulation's clock | simulation regularises when interpolation is easy, and bridges — modestly — only once it runs on the measurement's clock |

The post-competition study is complete: pre-registered predictions, every result table, the analysis scripts
(`*.py`, run from the repository root with the competition data at its default path, `data/RealPDE-Competition-Data`), and the RealPDEBench comparison as
a patch (`rb_streams.patch`) with its fetch and conversion scripts (`realpdebench/`), since competition rules keep
external data out of the training code.
