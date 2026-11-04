# Reproducing the submission (rewritten 2026-09-25 for the Decision Phase)

## One command
```bash
export REALPDE_DATA=/path/to/RealPDE-Competition-Data/training_data
bash tools/train_final.sh            # ≈ 3 h on one A100-PCIE-40GB incl. all checks (2 h 40 min measured 2026-09-18 with the
                                     # 4000-step student; the 12000-step student adds ≈ 15–45 min): 4 members → 6 teachers →
                                     # student → pack (with the fused-kernel cubins regenerated) → parity + recipe checks
python tools/check_recipe.py --pack submissions/rebuild/model.pth --replay-student   # recipe, provenance, bitwise tensors,
                                                                                     # kernel images, student replay
python tools/run_and_score.py submissions/rebuild/submission.py <eval_dir>          # packaging check (in-sample numbers)
```
The zip lands at `$REALISTIX_BUILD/rebuild.zip` (default `build`; set `REALISTIX_BUILD`
elsewhere). It contains `submission.py`, `model.pth` and two `fastgn_<sha12>_sm80/sm90.cubin` files.

**The method** (`docs/REPORT.md`): four forecasters (`plus_sim --film --cond norm`, width 96, mults 1,2,4, blocks 2,
loss `mse 1 + rel 8 + tke 1 + spec 0.1 + tmean 0.5 + sps 1 + mvpe 0.3`, Muon lr 0.02 / momentum 0.95 / wd 0.03,
EMA 0.998, 8000-step cosine read at step 6000, `--real-trajs all`, seeds 0 1 2 3); six interval-head *teachers*
(width 96, spread input, lr 8e-4, τ 0.05, 4000 steps, seeds 0 11 21 31 41 51) fitted on the four members; one
*student* of the same architecture regressed onto the teachers' averaged bounds (`tools/distil_head.py`: teacher-0
initialisation, under-width penalised 3×, 12000 steps, seed 7). Teachers are training artefacts and are not packed.
The inference source is `submissions/dec12kfinal5b/submission.py` (`CHAMPION` in `train_final.sh`): four members,
post-averaging gain dose 0.5, one head, fp16 channels_last CUDA graphs captured at import, and — inside the graphs
only — fused GroupNorm+SiLU / bias / head-input / output-assembly CUDA kernels whose source is embedded in `submission.py` (next section).

**Fused kernels.** The CUDA source is the string `_FGN_SRC` in `submission.py`. `train_final.sh` compiles it with
NVRTC (torch 2.2.2's own `nvidia-cuda-nvrtc` 12.1; no system compiler) into `fastgn_<sha12 of the source>_sm80.cubin`
and `_sm90.cubin` beside the copied `submission.py` (`tools/build_cubins.py`); NVRTC is deterministic, so
`python tools/build_cubins.py submissions/<name> --check` and `check_recipe` require the shipped cubins byte for
byte. At import the submission loads the cubin for its device's architecture through the CUDA driver API, else
compiles the same source with NVRTC, else captures its graphs with torch's native ops (same outputs to fp16 ulps,
slower); every captured bucket is checked against the native eager path first.

Knobs of `train_final.sh`: `GPU` (defaults to the inherited `CUDA_VISIBLE_DEVICES`, else 0), `MEMBERS` (default
`0 1 2 3`; the head's `--n-models` follows), `HEAD_MODE` (`student` default; `single` = one fitted head),
`TEACHER_SEEDS`, `HEAD_SEED` (single mode), `CHAMPION` (the inference source to ship; default `dec12kfinal5b`),
`NAME`/`SUB` (fresh run/submission names; the script refuses to reuse an existing run directory or overwrite a
`model.pth` unless `FORCE=1`), `R` / `REALISTIX_RUNS`, `PY`, `REALISTIX_BUILD`, `CHECK_PACK_EVAL` (cut from the release
with `tools/build_eval_npz.py` when absent), `W_MVPE` / `WD_MUON` / `LEAN` (experiment overrides). `MEMBER_RUNS` and
`TEACHER_RUNS` link existing checkpoints instead of training (audit/test only; a Decision-Phase rebuild sets neither).
`PRINT_RECIPE=1` prints the resolved argument lists and exits (it still creates the run root). The script refuses
duplicate seeds, an empty GPU mask, a card without CUDA bf16, and a release that does not resolve to exactly 81 real
(7575_0 excluded) + 100 simulated trajectories by count and hashed ordered manifest (`tools/release_manifest.json`);
it clears every `REALISTIX_HELDOUT_*` / `REALISTIX_SIM_*` research variable first.

## What the checks prove
* `tools/check_recipe.py` reads the pack's recorded provenance (member paths + SHA256, student path + SHA256, the
  student's teacher paths + SHA256 + seeds, member hashes recorded at every fit), re-hashes each file, compares every
  recorded `args` with the script's *resolved* literals (TRAIN for members, HEAD for teachers, STUDENT for the
  student), checks steps (members 6000, teachers 4000, student 12000), teacher count/distinctness/seeds, the
  member-hash chain (pack ↔ student ↔ every teacher), re-derives every packed tensor (folded, fp16) from its source
  and compares bitwise, checks the pack schema and the data manifest, requires `submission.py` to be the CHAMPION's
  source byte for byte, and requires the kernel cubins beside it to be a fresh NVRTC compile of its embedded source.
  `--replay-student` re-runs the distillation from the recorded teachers and requires the student bitwise — the only
  gate that tells a genuine 12000-step student from a snapshot or a substitute with copied metadata.
  `tools/tests/mutate_recipe.py` builds mutants (zeroed head, missing head tensor, flipped member tensor, a teacher
  substituted for the student, a step-2000 student, perturbed student weights with copied metadata, bad or duplicated
  teacher digests, five teachers, a student recording other members, a plain head where a student is required, a
  changed inference gain, corrupted / missing / stale kernel cubins) and asserts the checker rejects each and accepts
  the genuine pack (18/18 with the fused CHAMPION, 2026-09-24/25). Run it after every checker change.
* `tools/check_pack.py --eval` compares the packed `predict` (prediction and both bounds, guarded against the
  shipped gained forecast) against the members and student rebuilt in fp32 (tolerance 1e-3 relative; the pack reads
  ≈ 2e-4, the fp16 arithmetic).
* `tools/verify_best.py` proves the pinned zips on both read-only copies and each zip's `submission.py` against its
  packing commit.

## What is and is not reproducible
* **A pure `git archive` build (2026-09-24):** `git archive d490ad1` into an empty directory, no repository on the path,
  fresh roots, built on a different node and driver (A100-PCIE-40GB, driver 580.126.09 vs 595.71.05): members
  4 × 136/136, teachers 6 × 80/80, student 80/80 tensors bitwise identical to the deliverable; `check_pack` MATCH,
  `check_recipe` OK. The committed tree alone regenerates the shipped model.
* **Same card class and seeds: bit-identical end to end** — members, teachers and student, across A100-40GB nodes.
* **Different card class, same seeds: a different model.** No tensor identical; members as far apart as two seeds.
  Under bf16 autocast the seed pins the initialisation and the card class picks the trajectory. The organisers'
  retrain is therefore a fresh draw of the method (≈ ± 0.02 of final); four members, the distilled student and fixed
  steps are what lower its variance.
* **Inference repeatability.** The fused kernels are bitwise repeatable (fixed-order reductions, no atomics). torch's
  native transposed convolution (cuDNN dgrad) is not always: replays occasionally differ by one fp16 ulp on ≤ 2 % of
  elements, in every inference variant including the native one. Scores are unaffected.
* **fp16 packing.** The organisers discourage it; four fp32 members plus the head (≈ 414 MB) do not fit the 256 MB
  cap, and we chose four members over fp32 storage. All GPU arithmetic is fp16, so the stored tensors are exactly what
  executes; the one lossy step is the cast at packing (≈ 3e-6 relative L2; no subscore moves by 0.001).
* `7575_0` is a duplicated measurement withdrawn by the organisers and excluded in `realistix.splits`; its simulated
  trajectory stays in.

## Environment (pinned 2026-09-21; receipt `logs/env_pin.json`)
**python 3.10.20 · torch 2.2.2+cu121 (CUDA 12.1) · numpy 1.26.4 · h5py 3.16.0**, mirroring the evaluation image
`pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`; `requirements-training.txt`. The submission imports torch and numpy
(and the standard library's ctypes for the fused kernels); h5py is needed for training; NVRTC comes with torch's CUDA
libraries. A member peaks at 4.7 GiB of GPU memory in training.
