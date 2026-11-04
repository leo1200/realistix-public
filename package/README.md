# iapetos1918: Track 1 training package

This package retrains, from scratch, the method behind Codabench submission 947208. `METHOD.md` describes it. In
short: four U-Net forecasters are trained on mixed measured (`train_real/`) and simulated (`train_sim/`) windows; their
averaged forecast, with a fluctuation gain, is the prediction; six interval heads are fitted on a relaxed SPS reward and
distilled into one student head, which gives `lower` and `upper`.

## Codabench
- Username: iapetos1918
- Submission id this package reproduces: 947208
- final_score of that submission: 81.9960032201 (rel_l2 94.8462, tke 79.2327, mvpe 94.0911, sps 44.0018, time 93.5099
  = 3.511 ms per sample); rank 9 of 50 on the final development board; our last Development-Phase submission.

## How to reproduce
- Build command, if different from `docker build -t <name> .`: `docker build --platform linux/amd64 -t <name> .`, run
  inside the unzipped `realpde_t1_iapetos1918/` folder.
- Train command, if different from the default: the default. Evaluation: `load_predictor('/output', 'cuda')`, then
  `predict(x)`.

## Hardware your submitted model was trained on
- GPU model and count: 1 × NVIDIA A100-PCIE-40GB (driver 595.71.05).
- CPU: 2 × AMD EPYC 7452, 32 cores each (read from the node on 2026-10-01; the run did not log it); `OMP_NUM_THREADS=4`.
- Memory: 1007 GiB on the node; no limit was set on the job. Peak host memory of training: 3.4 GB (resident set).
- Training time on this hardware: 2 h 15 min 30 s (2026-09-23, 06:34:35–08:50:05 UTC): members 1 h 04 min (including
  about 3 min of preflight), teachers 48 min, student 23 min, packing and checks under 1 min.
- Peak GPU memory on this hardware: 7.8 GiB (7,983 MiB, whole device), reached in the parity check after packing;
  5.7 GiB while training the members, 3.6 GiB for the teachers, 4.2 GiB for the student. The 947208 run did not log
  memory; these are nvidia-smi `memory.used` readings every 5 s from a rehearsal of this package that reproduced the
  947208 model bit for bit on the same card class (see Notes).

## Environment
- Base image: `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`, pinned in the Dockerfile by its linux/amd64 digest
  `sha256:923f687790bec78081c357e71dcd5dcef80b0cc00f6c34484902a5e83362c854` (python 3.10.14, torch 2.2.2, CUDA 12.1,
  cuDNN 8.9.2, NVRTC 12.1.105, numpy 1.26.4).
- Added packages and versions (pinned the same way in the Dockerfile): `h5py==3.16.0`; `numpy==1.26.4` is the base
  image's own version, pinned so pip cannot move it. Nothing else: the fused CUDA kernels are compiled at training time
  by the base image's NVRTC, so no compiler is needed.
- The 947208 model itself was trained outside Docker, in a pip environment with the same package versions (torch
  2.2.2+cu121, numpy 1.26.4, h5py 3.16.0; python 3.10.20 where the image has 3.10.14); this image's software reproduces it bit for bit (see Notes).

## Data, weights and code
- Training data used: only `/data`. Members train on all 100 simulated and 81 measured trajectories, teachers and the
  student on the 81 measured ones. The withdrawn `7575_0` is excluded in code as well
  (`src/realistix/splits.py`). `mean_std_real.pt` is not read: each window is divided by its own RMS speed. Simulated
  pressure is not used.
- Augmentation: none. Windows of 20 input and 20 target frames are drawn uniformly over all (trajectory, start frame)
  positions; a member batch of 32 holds 13 measured and 19 simulated windows, the ratio of the two window counts.
- Official baseline checkpoints used, if any: none. Every checkpoint is produced by `train()`; the student starts from
  teacher 0 of the same run. The package contains no weight or data files.
- Third-party code and its license: the Muon optimizer in `src/realistix/muon.py`, adapted from
  https://github.com/KellerJordan/Muon (MIT; text in `THIRD_PARTY_NOTICES.md`); the organisers' scorer
  `starting_kit/scoring.py` from starting kit v9 (the organisers' code, redistributed unmodified; the kit ships no
  licence file), imported for `SIGMA_GLOBAL` and the metric functions the
  losses mirror; PyTorch, NumPy and h5py (BSD-3-Clause); CUDA libraries from the base image.

## Randomness
- Seed(s): members 0, 1, 2, 3; teachers 0, 11, 21, 31, 41, 51; student 7 (fixed in `tools/train_final.sh`).
- How runs or checkpoints are selected, if at all: nothing is selected inside `train()`. Step counts are fixed: each
  member keeps its EMA weights at step 6000 of an 8000-step cosine schedule, each teacher its last step (4000), the
  student its last step (12000). Every member and teacher is used. The scores printed during training are computed on
  trajectories the model trains on and drive no choice. The recipe constants themselves (step counts, member and
  teacher counts, loss weights, optimizer and head settings, gain dose) were chosen during development on folds of the
  training release that hold out whole (Re, angle) cells, and on development-board scores of our submissions.

## Track 2 only
- Does the method call the LLM gateway: not applicable (Track 1).

## Notes

**What `train()` does.** It copies the package to `/output/work` and runs our training script `tools/train_final.sh`
there, unchanged, with every output under `/output` (so the image's `/workspace` and the read-only `/data` are never
written by training). The script trains the members, the teachers and the student, packs them into `model.pth` (fp16),
compiles the two fused-kernel cubins from the CUDA source in the inference file, and runs our own checks (below).
`train()` then copies `submission.py` (947208's inference file, sha256 `1af16254833af11d3a7dfa02857801bf5b46aca380c3c7b726d735a8fcec1612`),
`model.pth` and `fastgn_93aac3a64160_sm80.cubin` / `_sm90.cubin` to the root of `/output`. `load_predictor` imports that
`submission.py` once per process; the import loads the model and captures its CUDA graphs. `predict` accepts any N
and returns float32 `prediction`, `lower` and `upper` of the input's shape (pressure channel 0).

```
/output/submission.py, model.pth, fastgn_*.cubin   what evaluation loads
/output/train.log, TRAIN_STATUS.json               the complete log; a summary with file hashes and check status
/output/runs/      members final_m0..3, teachers final_head_s*, student final_student (checkpoints, args, histories)
/output/build/     the packed Codabench-format directory and its zip;  /output/eval/  windows for the parity check
/output/work/      the package copy the script ran in
```
`/output` grows to about 4.6 GB. A previous run's entries, if any, are moved to `/output/_previous_<time>/`.

**Data identity.** The recipe records the release byte for byte (`tools/release_manifest.json`: names, order, sizes,
SHA256 of the 181 files). `train()` checks `/data` against it first. If anything differs it says so in `train.log`
and `TRAIN_STATUS.json` (`data_matches_release: false`) and trains on the provided data. The script requires 81
measured and 100 simulated trajectories.

**Our checks do not discard the model.** After packing, `tools/check_pack.py` compares the packed predictions and bounds
with the members and student rebuilt in fp32, and `tools/check_recipe.py` verifies every run's arguments against the
recipe, the checkpoint hash chain, every packed tensor bit for bit, the data manifest, the inference file and the cubins.
If a check fails, `train()` keeps the model, records the failure in `TRAIN_STATUS.json` and returns normally. It raises
only if no model was produced or the inference file is not 947208's.

**Rehearsals and expected spread.** We ran your training command's equivalent, `solution.train('/data', '/output',
'cuda')`, with this image's own Python stack (the base image's extracted Python, torch and CUDA libraries plus h5py
3.16.0) run on our host OS, not in a container (no container runtime was available to us): the package as the
working directory, a read-only copy of the release laid out as your `/data` (no `7575_0`, with `mean_std_real.pt`),
and HTTP(S) proxies pointed at a dead address (the code makes no network calls). On two shared A100-PCIE-40GB nodes (drivers 595.71.05 and 580.126.09; 2 h 22 min and
3 h 33 min) every check passed, and both times the resulting `model.pth` is **bit for bit the model of submission
947208**: all 624 packed tensors, and every member,
teacher and student checkpoint along the way (the files differ only in recorded metadata: paths, run names, file
times, the checkpoint hashes that cover those, and one new student argument that is off by default). On an H200 (a different
architecture) it took 1 h 36 min and passed every check; there the tensors differ, as expected across architectures,
and the in-sample scores stay close to 947208's model (rel 95.484 vs 95.483, sps 51.055 vs 51.050, tke 83.30 vs 83.34). The A800 shares the A100's architecture; we
could not test one, so we cannot say whether its libraries pick the same kernels. If they do not, the retrain is a
fresh draw of the method, about ±0.02 of final_score by our development rebuilds across card classes.
Training time on an A800: not measured; we expect it between the A100-PCIE-40GB and H200 figures above.

**Attached training log** (`logs/947208_training_run/`). `dec12k_rebuild.out` is the complete console log of the run
behind 947208 (job-queue lines are stamped in UTC+2), with its exit code, node record and driver script
(`rebuild_decision.sh`, which called `tools/train_final.sh` with the run name `dec12k`); `runs/` holds the members'
`args.json` and `history.json`, the teachers' `history.json` and `lam_scan.json` and the student's `history.json`. The
run used our repository at commit a49588c with uncommitted files in the working tree (the log's `tree clean: 125`
line counts `git status` entries), among them `tools/data_manifest.py`, `tools/release_manifest.json` and the driver
script, committed later; that the shipped code builds this model is shown by the A100 rehearsal above. Since then the
trainers changed only by one flag that is off by default; the build script and the checks gained guards, the kernel
compile and the kernel-image check, and new defaults. The run packed its model with an
earlier inference file (our submission 934147: CUDA graphs, no fused kernels); 947208 is the same `model.pth` with the
inference file shipped here. Metrics in the log are in-sample.

**Reading the code.** The shipped code is our development code unchanged. Its comments are development notes: some
cite notes that are not included, and some give earlier timings or earlier values of constants. The `TRAIN`, `HEAD`
and `STUDENT` argument lists in `tools/train_final.sh`, which the recipe check prints resolved in the log, are the
configuration used; other flags are ablation switches left at their defaults, and the code behind some of them
imports research modules that are not shipped and never run. In comments, "evaluation", "held-out" and val/test
mean holdouts cut from the training release.

**AI assistance.** The method was developed with AI coding assistants (Claude Code and OpenAI Codex), which wrote and
reviewed code and ran and analysed experiments under the author's direction. Code comments that mention "Codex", "agents", a review "round", a `claude/` research path or
"the owner" (the author) are notes from that process.
