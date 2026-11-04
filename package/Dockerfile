# RealPDE 2026 Track 1 (Sim2Real), team iapetos1918: training package reproducing Codabench submission 947208.
# The base is the organisers' template image, pinned by its linux/amd64 manifest digest (python 3.10.14, torch 2.2.2,
# CUDA 12.1, cuDNN 8.9.2, numpy 1.26.4, NVRTC 12.1.105 for the fused kernels). h5py is the one added package, pinned to
# the version the submitted model was trained with; numpy is pinned to the base image's own version so pip cannot move it.
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime@sha256:923f687790bec78081c357e71dcd5dcef80b0cc00f6c34484902a5e83362c854
RUN pip install --no-cache-dir h5py==3.16.0 numpy==1.26.4
WORKDIR /workspace
COPY . /workspace
