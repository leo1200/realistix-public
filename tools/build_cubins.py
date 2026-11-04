#!/usr/bin/env python3
"""Compile the fused-kernel source embedded in a submission.py into the cubins shipped beside it.

The fused GroupNorm+SiLU inference (dec12kgn*, 2026-09-24) carries its CUDA source as the `_FGN_SRC` string inside
submission.py and loads `fastgn_<sha12 of that source>_sm<arch>.cubin` from its own directory at import (else it compiles
the same source with NVRTC at import, else it captures its graphs with torch's native ops). The cubins are therefore
build artefacts of the shipped source, never an independent input: this tool regenerates them with the NVRTC that ships
with torch's CUDA libraries (nvidia-cuda-nvrtc, 12.1 for torch 2.2.2+cu121). NVRTC output is deterministic -- the same
source and version give byte-identical cubins, checked 2026-09-24 -- so `--check` can require the shipped files bitwise.

    python tools/build_cubins.py submissions/<name>            # write fastgn_<hash>_sm80/sm90.cubin
    python tools/build_cubins.py submissions/<name> --check    # exit 1 unless the shipped cubins equal a fresh compile

A submission.py without `_FGN_SRC` needs no cubins: the tool reports that and exits 0 (and --check requires that no
fastgn_*.cubin is present).
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import hashlib
import os
import re
import sys

ARCHES = ("80", "90")        # A100/A800 (the evaluation card), H100


def kernel_source(sub_py: str) -> str | None:
    m = re.search(r'^_FGN_SRC = r"""(.*?)"""', open(sub_py).read(), re.S | re.M)
    return m.group(1) if m else None


def _nvrtc():
    import torch  # noqa: F401 -- torch's own CUDA libraries are where nvrtc lives
    cands = []
    for p in sys.path:
        cands += glob.glob(os.path.join(p, "nvidia", "cuda_nvrtc", "lib", "libnvrtc.so*"))
    cands += glob.glob(os.path.join(sys.prefix, "lib", "libnvrtc.so*")) + ["libnvrtc.so.12", "libnvrtc.so"]
    for c in cands:
        if "builtins" in os.path.basename(c) or ".alt." in os.path.basename(c):
            continue
        try:
            return ctypes.CDLL(c)
        except OSError:
            continue
    raise SystemExit("libnvrtc not found (install torch's nvidia-cuda-nvrtc wheel)")


def compile_cubin(src: str, arch: str) -> bytes:
    """Byte-for-byte the compile the shipped submission performs as its own NVRTC fallback."""
    nv = _nvrtc()
    prog = ctypes.c_void_p()
    if nv.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"fastgn.cu", 0, None, None):
        raise SystemExit("nvrtcCreateProgram failed")
    opts = [f"--gpu-architecture=sm_{arch}".encode(), b"--std=c++14", b"-default-device"]
    if nv.nvrtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts)):
        n = ctypes.c_size_t(); nv.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
        log = ctypes.create_string_buffer(n.value); nv.nvrtcGetProgramLog(prog, log)
        raise SystemExit("nvrtcCompileProgram failed:\n" + log.value.decode(errors="replace"))
    n = ctypes.c_size_t(); nv.nvrtcGetCUBINSize(prog, ctypes.byref(n))
    buf = ctypes.create_string_buffer(n.value); nv.nvrtcGetCUBIN(prog, buf)
    nv.nvrtcDestroyProgram(ctypes.byref(prog))
    return buf.raw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sub_dir")
    ap.add_argument("--check", action="store_true", help="verify instead of write")
    a = ap.parse_args()
    sub_py = os.path.join(a.sub_dir, "submission.py")
    src = kernel_source(sub_py)
    shipped = sorted(glob.glob(os.path.join(a.sub_dir, "fastgn_*.cubin")))
    if src is None:
        if a.check and shipped:
            print(f"*** {sub_py} has no _FGN_SRC but {len(shipped)} fastgn cubin(s) sit beside it"); return 1
        print(f"{sub_py}: no fused-kernel source, no cubins needed"); return 0
    h = hashlib.sha256(src.encode()).hexdigest()[:12]
    want = {os.path.join(a.sub_dir, f"fastgn_{h}_sm{arch}.cubin"): compile_cubin(src, arch) for arch in ARCHES}
    if a.check:
        bad = [p for p in shipped if p not in want] + [p for p, img in want.items()
                                                        if not os.path.exists(p) or open(p, "rb").read() != img]
        for p in bad:
            print(f"*** {p}: missing, stale or not a fresh NVRTC compile of the shipped source")
        print(f"kernel source sha12 {h}: {len(want)} cubins {'OK' if not bad else 'FAIL'}")
        return 1 if bad else 0
    for p in shipped:
        if p not in want:
            os.remove(p)
    for p, img in want.items():
        open(p, "wb").write(img)
        print(f"wrote {p} ({len(img)} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
