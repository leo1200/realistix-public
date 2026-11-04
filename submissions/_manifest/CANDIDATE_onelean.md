# Candidate: onelean — one member, lean loss (2026-08-31)

**What it is.** The shipped 904950 recipe with two changes: a single width-96
member instead of three (`MEMBERS="0"`), and the lean loss (`LEAN=1`: w_mse,
w_tmean, w_spec to zero — individually dead, jointly +0.010 on 4/4 folds).
Head trained without the spread input (`--no-spread`; measures ~0 even with
three members). Everything else identical: plus_sim, film/norm folded at
export, Muon 0.02/0.95, EMA 0.998, rel 8 + tke 1 + sps 1, 8000-step cosine
taken at 6000 (lean fold vote 5000:5/6000:3, the same near-tie the board
tie-broke to 6000 twice), head 4000/4000 lr 8e-4 τ 0.05.

**Why one member is no longer a sacrifice.** Under w_tke 1.0 every member is
amplitude-calibrated, so ensemble averaging *over-damps*: the tke sign of
ensembling flipped (+0.5–0.6 tke for one member at every seed tried). The
2026-08 spread-input collapse (capture fraction 9% on ranking) removed the
other reason the ensemble existed.

**The gates, all pre-registered rigs:**

| gate | reads |
|---|---|
| hp_cur (3-member reference) | 84.886 |
| hp_one / one1 / one2 (single member, 3 seeds) | 85.078 ± 0.020 |
| hp_onelean (composed: single + lean) | **85.149** |

Composed weighted acc+sps delta vs hp_cur: rel −0.080, tke +0.566,
mvpe −0.037, sps −0.071 → **+0.000** (dead neutral). Time credit is
board-measured, not local: 893273 (same trunk shape, one pass) paid
**+1.065 time_score = +0.101 final** vs its three-pass sibling.

**Prediction, recorded before the score.** vs 904950 (81.597):
**+0.08 central, band [0.00, +0.15]** → board ≈ **81.68 [81.60, 81.75]**.
The haircut from +0.10 covers the tail→board transfer: mvpe transfers 5–7×
(893273's board mvpe read −0.226 under the old recipe; tail says −0.037 now)
and tke's sign is the least stable channel. Falsifier: a read below 81.60
means the tail gate's acc+sps neutrality does not transfer — the ensemble's
board value exceeds its tail value, and the single-member axis closes again
with a measured number.

**Artefact.** 71.4 MB fp16 pack (1×22.65M member + 13.09M head, 35.7M params;
zip 66.0 MB); one forward pass in the timed region. check_pack: **MATCH**
(max abs 1.28e-04 vs the run directory).

## Outcome (2026-09-01, submission 910411): 81.550 = −0.047. The falsifier tripped.

Below the band floor and the 81.60 line, so the pre-registered consequence
applies: **the single-member axis is closed with a board number — the
ensemble's board value exceeds its tail value.** The decomposition (board
delta / tail-gate delta): rel −0.087/−0.080 (1.1x, priced right),
tke +0.089/+0.566 (0.16x — the funding channel collapsed on hidden regimes),
sps −0.341/−0.071 (4.8x amplified), mvpe −0.010/−0.037, time +0.724 against
the +1.065 era measurement. Weighted sum −0.051 ≈ observed −0.047: the weight
fit is fine; two transfer coefficients were not.

The method lesson, recorded where the next gate will look: **a gate cleared by
tke funding is not cleared.** The composed gate's "dead neutral" was +0.566 of
tke covering −0.151 of rel+sps — and tke was already flagged the sign-unstable
channel. Under a tke-zeroed gate rule this candidate reads −0.059 weighted and
does not ship. Rank was never at risk (best-of; 904950 stands).
