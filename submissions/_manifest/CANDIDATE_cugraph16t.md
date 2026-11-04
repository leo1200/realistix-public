# Candidate: cugraph16t — fused GroupNorm+SiLU (Triton) on the fp16 graphs (2026-09-04)

**What it is.** cugraph16's `model.pth` byte-for-byte (md5 `137325e2…`) and its
`submission.py`, plus one fused kernel: every `SiLU(GroupNorm(·))` site in the members
and the head — 29 per member — runs as a single Triton launch (statistics, normalise,
affine, SiLU) instead of four. The compiled kernels ship in `triton_cache/` (1.1 MB, built
on an A100 with the image's torch 2.2.2 / Triton 2.2.0; the A800 has the same compute
capability 8.0), so the container never compiles; Triton's just-in-time path needs a host
C compiler the runtime image may not carry. A self-check at setup compares fused against
unfused output on a real-statistics window; any Triton failure, cache miss or self-check
miss leaves the unfused fp16 graphs (cugraph16) in charge.

**Why.** At one window per call the clock is the GPU-side chain: one fp16 channels_last
member is ~300 kernels for ~1.1 ms, 87 of them GroupNorm's three-kernel implementation
and 29 the SiLU after each. Fusing them: 293 → 131 kernels, 1.15 → 0.95 ms per member.

**Measured, one window per call, ms (cugraph16 → cugraph16t):**

| | N=1 | N=2 | N=64 |
|---|---|---|---|
| H100 | 3.13 → **2.35** (0.75×) | 3.63 → 2.90 | 37.3 → 26.3 |
| A100 (loaded node) | 3.61 → **3.13** (0.87×) | 4.38 → 3.62 | 61.0 → 43.5 |

Outputs within 2.4e-4 (N ≤ 64) / 4.9e-4 (N = 187) of cugraph16's. `run_and_score` on
`val_aoa` at chunk 64 and at chunk 1: rel_l2 95.474 / tke 83.272 / mvpe 95.815 / sps
51.141, identical to cugraph16 and 914270 to three decimals. Fresh process on the A100
with the shipped cache and **no compiler on the path**: setup 7.3–8.0 s (unchanged),
fused on, first call 4.6–4.9 ms at N=1; with an *empty* cache and no compiler: fused off,
prediction finite and correct — the fallback works. Zip `build/cugraph16t.zip`, 163.4 MB
extracted (35 files), smoke-tested standalone from a temp dir on both cards.

**Prediction, recorded before the score.** Reference is **cugraph16's 2026-09-05 read**
(expected ~4.9 ms/sample, `time_score` ~92.4). Scaling by the measured 0.75–0.87 gives
3.7–4.3 ms/sample → 93.3–92.8 → **+0.04 to +0.09 of final**. If the container's Triton
load fails, the fallback reproduces cugraph16 exactly (+0.00). **Central +0.06, band
[0.00, +0.12]** against cugraph16; against 914270 that is about +0.11 [0.00, +0.22].
Accuracy within **0.005** of 914270 on every channel.

**Falsifier.** `time_score` at or below cugraph16's with accuracy inside tolerance: either
the fallback fired (Triton unusable in the container — then a compiler-free path is not
available and kernel fusion closes) or the kernel chain is not where the container's
time is. Accuracy outside 0.005: revert to cugraph16.

**Armed 2026-09-04 for 2026-09-06T00:00:30Z.** Log `submit_cugraph16t.log`; zip md5s:
submission.py `54dce1b0…`, model.pth `137325e2…`.

## Prediction revised 2026-09-05 06:50Z, before the score, against the 09-05 read

cugraph16 (915299) read time 92.11282 = 5.34 ms/sample, a ratio of 0.94 to 914270
where the local GPU chain shrank 0.85–0.87. So the container's per-call time is
≈ 3.4 ms fixed + ≈ 2.0 ms of kernel chain, and the fused kernel's local −13 to −25 % on
the chain becomes −0.25 to −0.5 ms on the board: 5.34 → 4.85–5.1 ms/sample →
`time_score` 92.25–92.4 → **+0.014 to +0.03 of final**. **Revised central +0.02, band
[0.00, +0.05]** against 915299's 81.75365. The falsifier is unchanged: time at or below
92.113 with accuracy inside 0.005 means either the fallback fired or the chain is not
where the time is. Accuracy tolerance unchanged (within 0.005 of 914270).


## Scored 2026-09-06 (916264): **81.70813 = −0.046 against 915299; falsifier tripped**

| | 915299 cugraph16 | 916264 cugraph16t | Δ |
|---|---|---|---|
| rel_l2 / tke / mvpe / sps | 94.8238465339 / 79.0583912001 / 93.9105814514 / 43.7450757551 | **identical to ten decimals** | 0 |
| time_score | 92.11282 (5.34 ms/sample) | **91.64054 (6.07 ms/sample)** | −0.472 (+0.73 ms) |
| final | 81.75365 | 81.70813 | **−0.046** |

**The fused kernel did not run on the scored predictions.** The fused path differs from the torch path by up
to 2.4e-4 per element and locally moved the kit subscores in the third decimal; the board reproduces
cugraph16's subscores to ten decimals, which only the unfused graph path does. So the container took the
fallback — Triton absent/unusable, a cache-key miss with no compiler, or the self-check — exactly the case
the falsifier named. What the falsifier did not anticipate is a fallback that reads **slower** than the
artefact it falls back to: the timed path is then byte-for-byte cugraph16's, and locally (fresh process, H100,
N = 1) cugraph16 5.50 ms, cugraph16t with its shipped cache 4.31 ms, cugraph16t with an empty cache 4.30 ms
(a compiler exists on this host, so the compiler-less case cannot be reproduced here). The +0.73 ms is therefore
either platform drift on the day (the record's band is ±0.2 ms; 0.73 is outside it) or a container-side effect
of importing Triton / attempting a compile that we cannot observe. **Kernel fusion via Triton is closed for the
Development Phase**: the container will not run it, and the package cannot make it run. Anything shipped from
here uses cugraph16's inference code (`submissions/cugraph16/submission.py`, md5 249fd785…).

Consequence for mvpe03 (armed for 2026-09-07T00:00:30Z with the *same* submission.py, md5 54dce1b0…): its time
read would inherit this. Re-packed the same evening as `mvpe03b` = mvpe03's `model.pth` + cugraph16's
`submission.py`; see `CANDIDATE_mvpe03.md`.
