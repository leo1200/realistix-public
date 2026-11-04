# Joint Claude + Codex round, 2026-09-15 (slot proposal + breakthrough audit)

Brief: `brief_codex.md`. Codex (gpt-6-astra, xhigh) report: `codex/REPORT.md`, pick `codex/SLOT_PICK.md`,
registrations `codex/HEAD_PAIR_PREREG.md`, `codex/SPS_PREREG.md`.

## The board moved and STATE.md was stale
Live board 19:35 UTC: 1 np-user **81.85874** · 2 skabob 81.82371 · 3 roysegal 81.81876 · **4 us 81.80564** ·
10 modu-lemon 81.45764. STATE.md said "leader 81.824, np-user 81.807 at #3". **The gap to first is 0.0531, not
0.018**; the shortlist margin is 0.348. The 09-15 slot was never armed (`quota can=True`).

## Independent finding (Claude): the interval head's *policy* is at its optimum
`sps_ceiling.py` / `sps_probe2.py` on both local eval splits (187 and 175 windows, k4's own bounds):
W̄ = 0.6737 / 0.6727, coverage 0.884, corr(width, |err|) 0.652. Rescaling the head's bounds about the prediction has
its optimum at **c* = 1.000 exactly**; per-stratum rescaling on 400 strata (lead × width-decile × channel) buys
**+0.0000** on the held-out half; on 81,920 strata (y × x × lead × channel) it buys +0.0057 on the fit half and
**−0.0050** on the read half — pure overfitting. The realised-|error| oracle is 0.8416, unreachable without a better
forecaster. Codex reproduced this on three *held-out folds* with fresh outer-training-only heads and a 240-bin
empirical interval solver: every cross-fitted correction (global / lead / lead×channel×wake / width-bin scale+centre /
Gaussian / Laplace / empirical) is **negative on the fold mean**. The archive's "90–97 % of reachable gap" is now
measured for every rescaling family, and Codex is right that it still is not a full-information Bayes ceiling.
Correction to the arithmetic: the SPS prefactor is `Σ w·0.5/(0.5+e)` = **1/(1+2e)**, not the displayed `1/(1+e/2)`,
so R̄ = **0.716** (not 0.899) and board W̄ = **0.612** (not 0.488). The elasticity survives: +1 % relative on W̄ is
+0.438 SPS = **+0.111 final**.

## Codex's find, which Claude's pass missed: average the *outputs* of two independently fitted heads
Registered gate (mean held-out ΔSPS ≥ +0.08, every fold ≥ 0, no fold worse than either single head by 0.03) **passes**:
mim1 +0.0739 · far_drop +0.1252 · angle05 +0.1003, mean **+0.0998**, all 34 fold-cell readings positive. ≈ 2/3 of it is
interval centring, 1/3 width. Head *weight* soups, wider heads, head inputs/constants and OOF heads are closed; the
head's *output* was never ensembled. Point forecast bit-identical (verified: time-mean diff 0.0, variance ratio 1.0).
Cost: **+1.05 ms** locally (Claude, three interleaved cycles, fresh process, 300 N=1 calls: k4 4.175 / k4g 4.082 /
k4g10 4.111 / k4h2g05 5.157; Codex read +0.85). 233.6 MB of the 268.4 MB cap; 9/9 graphs; no fallback in 50 cases.

## Killed this round
Release-only covariance/convective augmentation (Codex, 12.9 CPU-seconds): pseudo-extrapolating the highest held-out
Re at four angles reduces TKE-pattern mismatch at only 1 of 4 angles (−10.2 %, −28.3 %, −7.2 %, **+10.8 %**). The
~18 A100-h ensemble arm is not funded. `codex/covariance_gate.json`.

## Arithmetic corrections to the record
* `CANDIDATE_k4g.md`'s half-dose break-even "5.32 ms" is wrong: the locked weights give **5.431 ms**, and the
  identical-zip repeat identifies the time weight as ≈ **0.09643**, giving N_half ≈ 0.00943 and break-even 5.414 ms.
* Dose scaling: doubling the dose multiplies the fold rel cost by 2.08–2.24 but the TKE benefit by only 1.62–1.70.
  Claude's "N ≈ 1.4–1.9 × the half-dose value" is optimistic; channel-wise transfer gives **N_full ≈ +0.012**, not
  +0.015–0.018. The cap binds on the same 0.2524 % of channel/cells at either dose.

## Candidates prepared (neither uploaded)
`k4h2g05.zip` sha256 `c743161a…` (recommended, `CANDIDATE_k4h2g05.md`) · `k4g10.zip` sha256 `50c40daf…`
(alternative, `CANDIDATE_k4g10.md`; built independently by both passes, byte-identical).
