# Candidate maskdistil — mask the unscored elements out of the distillation loss (pre-registered 2026-09-23 07:00 UTC, before any run)

**Owner's observation.** SPS does not score targets outside the PIV field of view or inside the airfoil body;
`starting_kit/scoring.py` drops every element whose target is **exactly zero**.

**What is true today.** The **teachers** honour that exactly — `train_final_head.py:485` builds
`scored = (y.abs() * sc > 0)` and `sps_reward` returns `(reward * scored).sum() / scored.sum()`, so no gradient ever
reaches an unscored element and whatever a teacher emits there is arbitrary (set by initialisation and the conv
net's spatial smoothness, nothing else). The **student does not**: `tools/distil_head.py` regresses onto the
teachers with `F.mse_loss(...)` / `d.pow(2).mean()` over **every** element. Measured on the release with the
shipped sampler, **10.12 % of elements are exactly zero** (per-batch 9.18–10.96 %). So about a tenth of the
distillation objective is spent imitating teacher output that carries no information about the metric — and the
asymmetric `under_weight` 3× penalty is applied there too, where "too narrow" cannot cost coverage because nothing
is scored.

**Arm.** `--mask-unscored` (added to `distil_head.py`, **off by default**): restrict both the symmetric and the
asymmetric branch to `scored`, built by the same expression the teachers use. Nothing else changes — same six
teachers, seed 7, teacher-0 init, under-width 3.0, width 96, 12000 steps, `--train-split train`.

**Gates, fixed now.**
* **Inertness (blocking).** With the flag absent, a rebuilt student must be **bitwise identical** to the one built
  before this edit on the same machine. Any difference kills the change outright, whatever the arm reads.
* **Read.** Fresh R3 joint geometry (15 held-out cells, 292 windows), `research/distil_2026-09-16/measure.py`,
  official SPS, gain 0.5, against the unmasked 12000-step student already measured there (46.9645).
* **Continue to an all-data build and a slot:** mean **ΔSPS ≥ +0.010** with **no cell < −0.030**, mirroring R4's
  registered floors. **Kill: mean ≤ 0, or any cell < −0.030.** Between: documented, no slot.
* The point forecast is untouched by construction, so rel / tke / mvpe must be identical to the digit; any movement
  quarantines the read.

**Prior.** Genuinely uncertain, and the honest case for a *small* effect is that the unscored region is spatially
contiguous and the head is a conv net: fitting arbitrary values there mainly costs capacity near the boundary,
which is where the wake — the part that matters — lives. The honest case for ~0 is that the teachers' output there
is smooth rather than adversarial, so imitating it is cheap. **Predicted +0.005 [−0.005, +0.020] SPS**; P(clears
+0.010) ≈ 0.3. This is a *specification* fix — the surrogate should not optimise what the metric ignores — so a
null result still leaves the masked form the better-specified one, but the registered floors decide the slot.

**Cost if adopted.** One flag in the contract (`STUDENT` gains `--mask-unscored`), no new stage, no size or latency
change, retrain cost unchanged.
