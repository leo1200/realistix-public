"""RealPDE Track 1 (Sim2Real) submission: a four-member U-Net ensemble with a post-averaging fluctuation gain and a
distilled, reward-trained interval head. torch + numpy (+ ctypes for the optional fused CUDA kernels); nothing else.

Trained by ``tools/train_final.sh`` on the 81 usable measured trajectories and the 100 simulated trajectories of the
official release, and on nothing else (``7575_0`` was withdrawn by the organisers on 2026-08-12 because it reuses
``6300_0``'s raw measurements). ``docs/REPORT.md`` describes the method; ``docs/MODEL.md`` carries the evidence ledger.

What happens per call (N windows of 20 input frames -> 20 predicted frames of u, v; p is returned as zeros):
  * Per-window normalisation: each window is divided by its own RMS speed over its valid (non-zero) cells, the forecast
    multiplied back. ``metadata`` is empty on scored calls and never read; no information crosses windows.
  * Four width-96 U-Nets (fp16, channels_last) predict the window; their plain mean is formed and each cell's temporal
    fluctuation about its time-mean is rescaled half-way toward the members' own average temporal variance (the gain,
    dose 0.5: target-free, and the time-mean -- hence MVPE -- is unchanged).
  * One interval head, reading the window, the plain mean and the members' per-element spread, returns the extents
    below and above the plain mean (a 12000-step distillation of six reward-trained heads); lower <= prediction <=
    upper is enforced, non-finite values are replaced, and all three arrays are returned as contiguous float32.
  * Every batch bucket (1..8, 16, 32, 64 windows) is captured as a CUDA graph at import -- the four members on four
    streams, then the head, gain and output assembly -- and replayed; page-locked copies return the outputs, the
    prediction's overlapping the head. If
    capture or its self-check against the eager path fails, the eager path runs instead.
  * Inside the captured graphs only, every GroupNorm+SiLU (and the convolution biases around them) runs as the custom
    CUDA kernels in ``_FGN_SRC`` below; the eager path is torch's native ops (see that section).
  * Setup is lean (one CPU thread with a GPU, CPU checkpoint load; each convolution's cuDNN algorithm is chosen by
    timing once at setup) and stderr carries a few FAQ-permitted diagnostics: import time, fused-kernel status,
    per-call time, page-fault delta and one SM-clock reading outside timed calls.
"""

from __future__ import annotations

import os
import sys
import time

# stderr-only diagnostics, which the FAQ explicitly allows (device, shapes, timings; never inputs or predictions).
# Outputs are untouched. They record how many fresh processes and calls a run makes and what each costs.
_T_IMPORT = time.perf_counter()
_T_EPOCH = time.time()
_NCALL = [0]


def _log(msg):
    try:
        sys.stderr.write(f"[rx {os.getpid()}] {msg}\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 -- diagnostics must never break inference
        pass


_DIAG = {}


def _diag(nvml=True):
    """(epoch s, SM clock MHz via NVML or -1, minor page faults) for the log. Never raises. NVML is read only when
    asked: at import end and after the untimed first call, never inside a call the harness times."""
    try:
        import resource
        flt = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    except Exception:  # noqa: BLE001
        flt = -1
    sm = -1
    try:
        if nvml and "nvml" not in _DIAG:
            # The NVML device of the CUDA device this process computes on, matched by PCI bus id (NVML does not
            # follow CUDA_VISIBLE_DEVICES; index 0 can be another job's GPU on a multi-GPU host).
            import ctypes as _ct
            _DIAG["nvml"] = None
            cu, lib = _ct.CDLL("libcuda.so.1"), _ct.CDLL("libnvidia-ml.so.1")
            dev, bus, h = _ct.c_int(), _ct.create_string_buffer(64), _ct.c_void_p()
            if (cu.cuDeviceGet(_ct.byref(dev), torch.cuda.current_device()) == 0
                    and cu.cuDeviceGetPCIBusId(bus, 64, dev) == 0 and lib.nvmlInit_v2() == 0
                    and lib.nvmlDeviceGetHandleByPciBusId_v2(bus, _ct.byref(h)) == 0):
                _DIAG["nvml"] = (lib, h)
        if nvml and _DIAG.get("nvml"):
            import ctypes as _ct
            lib, h = _DIAG["nvml"]
            c = _ct.c_uint()
            if lib.nvmlDeviceGetClockInfo(h, 1, _ct.byref(c)) == 0:     # NVML_CLOCK_SM
                sm = c.value
    except Exception:  # noqa: BLE001
        _DIAG["nvml"] = None
    return time.time(), sm, flt

import numpy as np
import torch
# Every tensor op runs on the GPU; a large CPU thread pool only costs setup time on a shared host.
if torch.cuda.is_available():
    torch.set_num_threads(1)
    # cuDNN picks each convolution's algorithm by timing the candidates once, when a bucket is first run (during the
    # untimed setup, before any graph is captured), restricted to deterministic algorithms; the captured graphs then
    # replay that choice. Measured 2026-09-25: -0.4 ms per 5-window call on an A100 against cuDNN's heuristic choice.
    # The chosen algorithm can differ between processes, so outputs can differ between processes by fp16 rounding.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
import torch.nn as nn
import torch.nn.functional as F

T_IN = T_OUT = 20
C = 2
EPS = 1e-8

_HERE = os.path.dirname(os.path.abspath(__file__))
_CKPT = os.path.join(_HERE, "model.pth")
#: Member-variance-matched fluctuation gain on the ensemble mean. Averaging
#: K members damps each cell's temporal fluctuation by the members' disagreement; the
#: scorer's TKE pays for that damping. Per window and cell the mean's temporal variance is
#: rescaled toward the members' own average temporal variance -- a quantity the members
#: supply, no target involved -- at dose GAIN_DOSE (0 = the plain mean, 1 = full match).
#: The time-mean of every cell is unchanged, so MVPE is untouched; the interval head still
#: reads and brackets the plain mean (the lower <= prediction <= upper guard below may widen a few elements where the
#: gained prediction leaves them).
GAIN_DOSE = 0.5
GAIN_CAP = 3.0


def _matched_gain(outs, y):
    """outs (K, B, T*C, H, W) members, y (B, T*C, H, W) their mean, both in normalised units.
    Returns y with each (window, channel, cell) fluctuation about its time-mean scaled by
    1 + dose * (sqrt(v_members / v_mean) - 1), capped."""
    k, b, tc, h, w = outs.shape
    o = outs.reshape(k, b, T_OUT, C, h, w).float()
    ym = y.reshape(b, T_OUT, C, h, w).float()
    mu = ym.mean(dim=1, keepdim=True)
    v_mean = ((ym - mu) ** 2).mean(dim=1, keepdim=True)
    v_mem = ((o - o.mean(dim=2, keepdim=True)) ** 2).mean(dim=2, keepdim=True).mean(dim=0)
    g = torch.sqrt(v_mem / v_mean.clamp_min(1e-12)).clamp(max=GAIN_CAP)
    g = 1.0 + GAIN_DOSE * (g - 1.0)
    return (mu + g * (ym - mu)).reshape(b, tc, h, w).to(y.dtype)
_BATCH = 256


# ---------------------------------------------------------------------------------------------------
# Fused GroupNorm + SiLU (and the convolution biases around it) for the captured graphs.
#
# torch 2.2's native GroupNorm on a channels_last fp16 tensor costs 66-77 us at 5x96x32x64 on an A100 while
# the 3x3 conv beside it costs 17 us, and every GroupNorm in both networks is followed by a SiLU: ~29 such
# pairs per member and ~15 in the interval head were about half of the timed GPU chain. One kernel replaces each
# pair: one block per (window, group), fp32 shifted moments with fixed-order reductions (bitwise repeatable), then
# the per-channel scale / shift and the SiLU in one pass with one fp16 rounding (closer to an fp32 reference than
# the native pair, which rounds twice). The convolutions around it run bias-free and their biases are folded in:
# c1's into the next GroupNorm, c2's and the skip's into one residual add, the output conv's into the last-frame
# anchor (members) or the head's clamp + exp. In the parallel buckets' post graph two more kernels replace the ~40
# small ones between the members and the output: the head's input (mean, spread) and the gain + output assembly.
#
# No compiler is needed where this runs: the archive ships the cubins that NVRTC 12.1 (torch 2.2.2's own
# nvidia-cuda-nvrtc) compiled from exactly the source below, named by its sha256 (tools/build_cubins.py regenerates
# and verifies them; NVRTC is deterministic); without a matching cubin the source is compiled with NVRTC at import.
# Kernels are launched through the CUDA driver API on torch's current stream, so the launches are captured into the
# CUDA graphs exactly like torch's own kernels. They are used only while building graphs (the uncaptured warm run
# and the capture); the eager path is torch's native ops, and every captured bucket is checked against it on
# synthetic data at import. If the kernels cannot be loaded, or a bucket's check or capture raises, that bucket and
# the later ones are captured with the native ops (the import log's fb= counts the fused buckets).
# ---------------------------------------------------------------------------------------------------
_FUSED = [False]          # True only while capturing graphs with the fused kernels loaded


def _norm_act(gn, x, pre_bias=None):
    """silu(gn(x + pre_bias)); pre_bias is an fp32 per-channel vector (the preceding conv's bias) or None."""
    if _FUSED[0] and _fgn_supported(x, gn):
        return _fgn_gn_silu(x, gn) if pre_bias is None else _fgn_gn_silu(x, gn, pre_bias=pre_bias)
    if pre_bias is not None:
        x = x + pre_bias.to(x.dtype)[None, :, None, None]
    return F.silu(gn(x))


def _fgn_f32(mod, name):
    """An fp32 copy of a module's parameter, cached on the module (built outside capture, on the first warm run)."""
    key = "_fgn_" + name + "32"
    t = mod.__dict__.get(key)
    if t is None:
        t = getattr(mod, name).detach().float().contiguous()
        mod.__dict__[key] = t
    return t


def _fgn_bsum(blk):
    """c2's bias plus the 1x1 skip's bias (if any), fp32, cached on the block."""
    t = blk.__dict__.get("_fgn_bsum32")
    if t is None:
        t = blk.c2.bias.detach().float()
        if not isinstance(blk.skip, nn.Identity):
            t = t + blk.skip.bias.detach().float()
        t = t.contiguous()
        blk.__dict__["_fgn_bsum32"] = t
    return t



import ctypes
import glob
import hashlib

_FGN_SRC = r"""
typedef unsigned short h16;
__device__ __forceinline__ float h2f(h16 h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
__device__ __forceinline__ h16 f2h(float f) { h16 h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }

// GroupNorm (+ SiLU) of x + pbias (pbias: fp32 per channel, or null).  grid (G, B), block T threads with T a multiple of 32 and of cq = Cg/4.
// Thread t owns the 4 channels q = t % cq (8-byte loads, no per-element division) of pixels
// p = t / cq, t / cq + T / cq, ...; loads are issued 4 at a time before use.  Fixed-order reductions.
extern "C" __global__ void gn_group3(const h16* __restrict__ x, const float* __restrict__ pbias,
                                     const h16* __restrict__ gamma, const h16* __restrict__ beta,
                                     h16* __restrict__ y, int C, int HW, int G, float eps, int silu) {
    __shared__ float red[64];
    const int g = blockIdx.x, b = blockIdx.y, t = threadIdx.x, T = blockDim.x;
    const int cg = C / G, cq = cg >> 2;
    const int q = t % cq, p0 = t / cq, ps = T / cq;
    const size_t base = (size_t)b * HW * C + (size_t)g * cg + 4 * q;
    float pb[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) pb[j] = pbias ? pbias[g * cg + 4 * q + j] : 0.f;
    const float K = h2f(x[(size_t)b * HW * C + (size_t)g * cg]) + (pbias ? pbias[g * cg] : 0.f);
    float s1 = 0.f, s2 = 0.f;
    int p = p0;
    for (; p + 3 * ps < HW; p += 4 * ps) {
        uint2 v[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) v[u] = __ldg(reinterpret_cast<const uint2*>(x + base + (size_t)(p + u * ps) * C));
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const h16* e = reinterpret_cast<const h16*>(&v[u]);
            #pragma unroll
            for (int j = 0; j < 4; ++j) { float d = h2f(e[j]) + pb[j] - K; s1 += d; s2 += d * d; }
        }
    }
    for (; p < HW; p += ps) {
        uint2 v = __ldg(reinterpret_cast<const uint2*>(x + base + (size_t)p * C));
        const h16* e = reinterpret_cast<const h16*>(&v);
        #pragma unroll
        for (int j = 0; j < 4; ++j) { float d = h2f(e[j]) + pb[j] - K; s1 += d; s2 += d * d; }
    }
    for (int o = 16; o > 0; o >>= 1) { s1 += __shfl_xor_sync(0xffffffff, s1, o); s2 += __shfl_xor_sync(0xffffffff, s2, o); }
    const int w = t >> 5, l = t & 31, nw = T >> 5;
    if (l == 0) { red[w] = s1; red[32 + w] = s2; }
    __syncthreads();
    if (t < 32) {
        s1 = t < nw ? red[t] : 0.f; s2 = t < nw ? red[32 + t] : 0.f;
        for (int o = 16; o > 0; o >>= 1) { s1 += __shfl_xor_sync(0xffffffff, s1, o); s2 += __shfl_xor_sync(0xffffffff, s2, o); }
        __syncwarp();   // every lane has read red[t] before lane 0 reuses red[0..1] (shuffles are not memory barriers)
        if (t == 0) {
            const float cnt = (float)HW * (float)cg;
            const float m = s1 / cnt;
            float var = s2 / cnt - m * m; var = var > 0.f ? var : 0.f;
            red[0] = K + m; red[1] = rsqrtf(var + eps);
        }
    }
    __syncthreads();
    const float mean = red[0], rstd = red[1];
    float a[4], d[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        a[j] = rstd * h2f(gamma[g * cg + 4 * q + j]);
        d[j] = h2f(beta[g * cg + 4 * q + j]) - mean * a[j];
    }
    for (p = p0; p < HW; p += ps) {
        const size_t off = base + (size_t)p * C;
        uint2 v = __ldg(reinterpret_cast<const uint2*>(x + off));
        h16* e = reinterpret_cast<h16*>(&v);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float z = (h2f(e[j]) + pb[j]) * a[j] + d[j];
            if (silu) z = z / (1.f + expf(-z));
            e[j] = f2h(z);
        }
        *reinterpret_cast<uint2*>(y + off) = v;
    }
}

// out = a + r + bias_c  (r may be null), fp16 NHWC with C % 8 == 0, 8 halves per thread; out may alias a,
// so a is neither __restrict__ nor read through the read-only cache.
extern "C" __global__ void add_bias_res(const h16* a, const h16* __restrict__ r,
                                        const float* __restrict__ bias, h16* out, int C, long long nvec) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nvec) return;
    const int cb = (int)((i * 8) % C);
    uint4 qa = reinterpret_cast<const uint4*>(a)[i];      // a may alias out (in place): ordinary read
    uint4 qr; if (r) qr = __ldg(reinterpret_cast<const uint4*>(r) + i);
    h16* ea = reinterpret_cast<h16*>(&qa); const h16* er = reinterpret_cast<const h16*>(&qr);
    #pragma unroll
    for (int j = 0; j < 8; ++j) ea[j] = f2h(h2f(ea[j]) + (r ? h2f(er[j]) : 0.f) + bias[cb + j]);
    reinterpret_cast<uint4*>(out)[i] = qa;
}

// y += bias_c + x[pixel, off + c % per]   (the U-Net's last-frame anchor), y: C channels, x: Cx channels, NHWC.
extern "C" __global__ void add_bias_last(h16* y, const float* __restrict__ bias, const h16* __restrict__ x,
                                         int C, int Cx, int off, int per, long long nvec) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nvec) return;
    const long long pix = (i * 8) / C;
    const int cb = (int)((i * 8) % C);
    uint4 q = reinterpret_cast<uint4*>(y)[i];
    h16* e = reinterpret_cast<h16*>(&q);
    #pragma unroll
    for (int j = 0; j < 8; ++j) e[j] = f2h(h2f(e[j]) + bias[cb + j] + h2f(x[pix * Cx + off + (cb + j) % per]));
    reinterpret_cast<uint4*>(y)[i] = q;
}

// y = exp(clamp(y + bias_c, lo, hi)) in place (the interval head's width output).
extern "C" __global__ void bias_clamp_exp(h16* y, const float* __restrict__ bias, int C, float lo, float hi,
                                          long long nvec) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nvec) return;
    const int cb = (int)((i * 8) % C);
    uint4 q = reinterpret_cast<uint4*>(y)[i];
    h16* e = reinterpret_cast<h16*>(&q);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float z = h2f(e[j]) + bias[cb + j];
        z = z < lo ? lo : (z > hi ? hi : z);
        e[j] = f2h(expf(z));
    }
    reinterpret_cast<uint4*>(y)[i] = q;
}
// ---- glue kernels (v4): the post graph's mean / spread / head input and gain / assembly, each in one pass ----------
// Head input, NHWC (B, H, W, 161) fp16: [0:40) the normalised window, [40:80) the members' plain mean, [80] log(scale),
// [81:121) the spread r = std_unbiased * scale / sigma_global, [121:161) log(max(r, 1e-4)).  One thread per
// (pixel, channel < 40).  Orders follow torch's: mean = (((o0+o1)+o2)+o3) * 0.25, std by sequential Welford.
extern "C" __global__ void head_prologue(const h16* __restrict__ m0, const h16* __restrict__ m1,
                                         const h16* __restrict__ m2, const h16* __restrict__ m3,
                                         const h16* __restrict__ xin, const float* __restrict__ scale,
                                         h16* __restrict__ z, int HW, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long pix = i / 40; const int c = (int)(i - pix * 40);
    const int b = (int)(pix / HW);
    const float o[4] = {h2f(m0[i]), h2f(m1[i]), h2f(m2[i]), h2f(m3[i])};
    const float mean = (((o[0] + o[1]) + o[2]) + o[3]) * 0.25f;
    float wm = 0.f, m2acc = 0.f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) { const float d = o[k] - wm; wm += d / (float)(k + 1); m2acc += d * (o[k] - wm); }
    const float sd = sqrtf(m2acc / 3.f);
    const float s = scale[b];
    const float r = (sd * s) / 0.0563870259f;
    h16* zp = z + pix * 161;
    zp[c] = xin[i];
    zp[40 + c] = f2h(mean);
    zp[81 + c] = f2h(r);
    zp[121 + c] = f2h(logf(r < 1e-4f ? 1e-4f : r));
    if (c == 0) zp[80] = f2h(logf(s));
}

__device__ __forceinline__ float fin16(float v) { return (isnan(v) || isinf(v)) ? 0.f : v; }

// Gain + assembly: out (3, B, T=20, HW, 3) fp32 = [prediction, lower, upper], each the fp16 value widened.  One thread
// per (window, pixel, velocity channel c < 2); member channel t*2 + c; d is the head output (B, H, W, 80) NHWC (below
// in [0:40), above in [40:80)).  pred = fp16(gained mean * s), lower/upper = fp16(plain mean * s -/+ width), non-finite
// -> 0, then lower <= pred <= upper; the pressure channel is written as 0.
extern "C" __global__ void post_assemble(const h16* __restrict__ m0, const h16* __restrict__ m1,
                                         const h16* __restrict__ m2, const h16* __restrict__ m3,
                                         const h16* __restrict__ d, const float* __restrict__ scale,
                                         float* __restrict__ out, int HW, int B, float dose, float cap, int part) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * HW * 2) return;
    const int c = i & 1, pix = i >> 1, b = pix / HW, p = pix - b * HW;
    const h16* ms[4] = {m0, m1, m2, m3};
    const size_t base = (size_t)pix * 40 + c;
    float y[20], mk[4] = {0.f, 0.f, 0.f, 0.f};
    float mu = 0.f;
    #pragma unroll
    for (int t = 0; t < 20; ++t) {
        float o[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) { o[k] = h2f(ms[k][base + 2 * t]); mk[k] += o[k]; }
        y[t] = (((o[0] + o[1]) + o[2]) + o[3]) * 0.25f;
        mu += y[t];
    }
    mu *= 0.05f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) mk[k] *= 0.05f;
    float vmean = 0.f, vk[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int t = 0; t < 20; ++t) {
        const float e = y[t] - mu; vmean += e * e;
        #pragma unroll
        for (int k = 0; k < 4; ++k) { const float f = h2f(ms[k][base + 2 * t]) - mk[k]; vk[k] += f * f; }
    }
    vmean *= 0.05f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) vk[k] *= 0.05f;
    const float vmem = (((vk[0] + vk[1]) + vk[2]) + vk[3]) * 0.25f;
    float g = sqrtf(vmem / (vmean < 1e-12f ? 1e-12f : vmean));
    g = g > cap ? cap : g;
    g = 1.f + dose * (g - 1.f);
    const float s = scale[b];
    const size_t plane = (size_t)B * 20 * HW * 3;
    for (int t = 0; t < 20; ++t) {
        const float yg = mu + g * (y[t] - mu);
        const float yp = y[t] * s;
        const float pr = fin16(h2f(f2h(yg * s)));
        const size_t o = (((size_t)b * 20 + t) * HW + p) * 3;
        if (part != 1) {                    // the prediction plane (needs no widths)
            out[o + c] = pr;
            if (c == 0) out[o + 2] = 0.f;
        }
        if (part != 0) {                    // the bounds, contained around the same prediction
            float lo = fin16(h2f(f2h(yp - h2f(d[(size_t)pix * 80 + 2 * t + c]))));
            float hi = fin16(h2f(f2h(yp + h2f(d[(size_t)pix * 80 + 40 + 2 * t + c]))));
            lo = lo < pr ? lo : pr;
            hi = hi > pr ? hi : pr;
            out[plane + o + c] = lo; out[2 * plane + o + c] = hi;
            if (c == 0) { out[plane + o + 2] = 0.f; out[2 * plane + o + 2] = 0.f; }
        }
    }
}
"""

_FGN = {}


def _fgn_nvrtc():
    """Locate libnvrtc: torch's pip dependency, the conda prefix, then the loader path."""
    cands = []
    for p in sys.path:
        cands += glob.glob(os.path.join(p, "nvidia", "cuda_nvrtc", "lib", "libnvrtc.so*"))
    for root in {sys.prefix, os.path.dirname(os.path.dirname(torch.__file__))}:
        cands += glob.glob(os.path.join(root, "lib", "libnvrtc.so*"))
    cands += glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libnvrtc*.so*"))
    cands += ["libnvrtc.so.12", "libnvrtc.so"]
    for c in cands:
        if "builtins" in os.path.basename(c) or ".alt." in os.path.basename(c):
            continue
        try:
            return ctypes.CDLL(c)
        except OSError:
            continue
    return None


def _fgn_compile(arch):
    """NVRTC-compile _FGN_SRC for sm_<arch>; returns cubin bytes or raises."""
    nv = _fgn_nvrtc()
    if nv is None:
        raise RuntimeError("libnvrtc not found")
    prog = ctypes.c_void_p()
    r = nv.nvrtcCreateProgram(ctypes.byref(prog), _FGN_SRC.encode(), b"fastgn.cu", 0, None, None)
    if r:
        raise RuntimeError(f"nvrtcCreateProgram {r}")
    opts = [f"--gpu-architecture=sm_{arch}".encode(), b"--std=c++14", b"-default-device"]
    arr = (ctypes.c_char_p * len(opts))(*opts)
    r = nv.nvrtcCompileProgram(prog, len(opts), arr)
    if r:
        n = ctypes.c_size_t()
        nv.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
        buf = ctypes.create_string_buffer(n.value)
        nv.nvrtcGetProgramLog(prog, buf)
        raise RuntimeError(f"nvrtcCompileProgram {r}: {buf.value.decode(errors='replace')[:2000]}")
    n = ctypes.c_size_t()
    if nv.nvrtcGetCUBINSize(prog, ctypes.byref(n)):
        raise RuntimeError("nvrtcGetCUBINSize")
    buf = ctypes.create_string_buffer(n.value)
    if nv.nvrtcGetCUBIN(prog, buf):
        raise RuntimeError("nvrtcGetCUBIN")
    nv.nvrtcDestroyProgram(ctypes.byref(prog))
    return buf.raw


def _fgn_src_hash():
    return hashlib.sha256(_FGN_SRC.encode()).hexdigest()[:12]


def _fgn_load(cubin_dir=None):
    """Load the kernels into torch's current CUDA context. Returns True on success.

    A cubin compiled from this exact source (file name carries the source hash and the architecture) is
    used when shipped next to this file -- loading it costs ~1 ms where NVRTC costs ~0.25 s per fresh
    process; otherwise the source is compiled with NVRTC for the device's architecture."""
    if _FGN:
        return True
    torch.cuda.init()
    torch.zeros(1, device="cuda")          # make sure the primary context is current on this thread
    maj, mnr = torch.cuda.get_device_capability()
    arch = f"{maj}{mnr}"
    cu = ctypes.CDLL("libcuda.so.1")
    mod, err, origin = ctypes.c_void_p(), [], None
    images = []
    if cubin_dir:
        p = os.path.join(cubin_dir, f"fastgn_{_fgn_src_hash()}_sm{arch}.cubin")
        if os.path.exists(p):
            images.append(("cubin", lambda p=p: open(p, "rb").read()))
    images.append(("nvrtc", lambda: _fgn_compile(arch)))
    for name, get in images:
        try:
            image = get()
            r = cu.cuModuleLoadData(ctypes.byref(mod), ctypes.c_char_p(image))
            if r:
                raise RuntimeError(f"cuModuleLoadData {r}")
            origin = name
            break
        except Exception as e:  # noqa: BLE001
            err.append(f"{name}: {e!r}")
    if origin is None:
        raise RuntimeError("no kernel image: " + "; ".join(err))
    fns = {}
    for name in ("gn_group3", "add_bias_res", "add_bias_last", "bias_clamp_exp", "head_prologue", "post_assemble"):
        fns[name] = ctypes.c_void_p()
        if cu.cuModuleGetFunction(ctypes.byref(fns[name]), mod, name.encode()):
            raise RuntimeError(f"cuModuleGetFunction {name}")
    cu.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
    _FGN.update(cu=cu, mod=mod, fns=fns, arch=arch, origin=origin)
    return True


def _fgn_launch(f, grid, block, shmem, args):
    cargs = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    stream = torch.cuda.current_stream().cuda_stream
    r = _FGN["cu"].cuLaunchKernel(f, grid[0], grid[1], 1, block, 1, 1, shmem, ctypes.c_void_p(stream), cargs, None)
    if r:
        raise RuntimeError(f"cuLaunchKernel {r}")


def _fgn_ptr(t):
    return ctypes.c_void_p(t.data_ptr() if t is not None else 0)


def _fgn_gn_silu(x, gn, silu=True, threads=None, pre_bias=None):
    """silu(group_norm(x)) for fp16 channels_last x (B, C, H, W) with nn.GroupNorm `gn` (fp16 affine).
    Threads: a multiple of 96 (so of 32 and of every Cg/4 used here), ~4 vectors each, at most 960."""
    B, C, H, W = x.shape
    G = gn.num_groups
    cq = (C // G) // 4
    if threads is None:
        nvec = H * W * cq
        threads = min(960, max(96, 96 * -(-nvec // (96 * 4))))
    y = torch.empty_like(x, memory_format=torch.channels_last)
    _fgn_launch(_FGN["fns"]["gn_group3"], (G, B), threads, 0,
            [ctypes.c_void_p(x.data_ptr()), _fgn_ptr(pre_bias), ctypes.c_void_p(gn.weight.data_ptr()),
             ctypes.c_void_p(gn.bias.data_ptr()), ctypes.c_void_p(y.data_ptr()), ctypes.c_int(C),
             ctypes.c_int(H * W), ctypes.c_int(G), ctypes.c_float(gn.eps), ctypes.c_int(1 if silu else 0)])
    return y


def _fgn_vec_launch(name, nvec, args):
    T = 256
    _fgn_launch(_FGN["fns"][name], (-(-nvec // T), 1), T, 0, args + [ctypes.c_longlong(nvec)])


def _fgn_add_bias(a, r, bias):
    """a + r + bias (per channel, fp32), in place into a (a fresh conv output). r may be None."""
    _fgn_vec_launch("add_bias_res", a.numel() // 8, [_fgn_ptr(a), _fgn_ptr(r), _fgn_ptr(bias), _fgn_ptr(a), ctypes.c_int(a.shape[1])])
    return a


def _fgn_add_last_bias(y, bias, x, off, per):
    """y += bias + x[:, off + c % per] per pixel (the last-frame anchor), in place."""
    _fgn_vec_launch("add_bias_last", y.numel() // 8, [_fgn_ptr(y), _fgn_ptr(bias), _fgn_ptr(x), ctypes.c_int(y.shape[1]),
                                                   ctypes.c_int(x.shape[1]), ctypes.c_int(off), ctypes.c_int(per)])
    return y


def _fgn_clamp_exp_bias(y, bias, lo, hi):
    _fgn_vec_launch("bias_clamp_exp", y.numel() // 8, [_fgn_ptr(y), _fgn_ptr(bias), ctypes.c_int(y.shape[1]),
                                                    ctypes.c_float(lo), ctypes.c_float(hi)])
    return y


def _fgn_head_input(ms, xin, scale):
    """The interval head's 161-channel input (window, plain mean, log scale, spread, log spread) in one kernel."""
    B, _, H, W = xin.shape
    z = torch.empty((B, 161, H, W), dtype=torch.float16, device=xin.device, memory_format=torch.channels_last)
    n = B * H * W * 40
    _fgn_launch(_FGN["fns"]["head_prologue"], (-(-n // 256), 1), 256, 0,
                [_fgn_ptr(ms[0]), _fgn_ptr(ms[1]), _fgn_ptr(ms[2]), _fgn_ptr(ms[3]), _fgn_ptr(xin), _fgn_ptr(scale),
                 _fgn_ptr(z), ctypes.c_int(H * W), ctypes.c_longlong(n)])
    return z


def _fgn_assemble(ms, d, scale, part=2, o=None):
    """(3, B, 20, H, W, 3) float32 [prediction, lower, upper] -- gain, scaling, bounds, guards -- in one kernel.
    part 0 writes the prediction plane only (d unused, may be None), 1 the bounds only (into o), 2 all three."""
    B, _, H, W = ms[0].shape
    if o is None:
        o = torch.empty((3, B, T_OUT, H, W, 3), dtype=torch.float32, device=ms[0].device)
    n = B * H * W * 2
    _fgn_launch(_FGN["fns"]["post_assemble"], (-(-n // 128), 1), 128, 0,
                [_fgn_ptr(ms[0]), _fgn_ptr(ms[1]), _fgn_ptr(ms[2]), _fgn_ptr(ms[3]), _fgn_ptr(d), _fgn_ptr(scale),
                 _fgn_ptr(o), ctypes.c_int(H * W), ctypes.c_int(B), ctypes.c_float(GAIN_DOSE), ctypes.c_float(GAIN_CAP),
                 ctypes.c_int(part)])
    return o


def _fgn_nhwc16(*ts):
    """Every tensor fp16, CUDA, channels_last-contiguous, 16-byte aligned, channels % 8 == 0 (None allowed)."""
    return bool(_FGN) and all(t is None or (t.is_cuda and t.dtype == torch.float16 and t.dim() == 4
                                            and t.is_contiguous(memory_format=torch.channels_last)
                                            and t.data_ptr() % 16 == 0 and t.shape[1] % 8 == 0) for t in ts)


def _fgn_supported(x, gn):
    if not (_FGN and x.is_cuda and x.dtype == torch.float16 and x.dim() == 4 and gn.affine
            and gn.weight.dtype == torch.float16 and x.is_contiguous(memory_format=torch.channels_last)
            and x.data_ptr() % 16 == 0 and x.shape[1] % gn.num_groups == 0):
        return False
    cg = x.shape[1] // gn.num_groups
    return cg % 4 == 0 and 96 % (cg // 4) == 0


class Block(nn.Module):
    """With ``n_modalities`` the block carries a per-domain affine at each norm.

    The convolutions stay shared, so simulated windows still teach the real branch
    its dynamics, while the statistics each domain is normalised to may differ.
    (Worth +0.17 under the 2026-08 MSE-era loss; re-measured 2026-08-31 under the
    final recipe the conditioning's marginal is inside noise -- the modality is
    perfectly separable from the input via the optical shadow, and the rel-
    dominated loss removed the averaging conflict. Kept: folds away at export.)
    The original placement argument, retained for the record:
    the sim/real gap is a fluctuation-energy gap, and energy is set where the
    network produces output, not where it reads input. Inference always selects
    the real row.
    """

    def __init__(self, cin, cout, groups=8, n_modalities=0):
        super().__init__()
        self.n1 = nn.GroupNorm(min(groups, cin), cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(min(groups, cout), cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.n_modalities = n_modalities
        if n_modalities:
            self.f1 = nn.Embedding(n_modalities, 2 * cin)
            self.f2 = nn.Embedding(n_modalities, 2 * cout)

    def _mod(self, emb, h, mod):
        g, b = emb(mod).chunk(2, dim=-1)
        return h * (1 + g[:, :, None, None]) + b[:, :, None, None]

    def forward(self, x, mod=None):
        if not (self.n_modalities and mod is not None):
            if _FUSED[0] and _fgn_nhwc16(x) and _fgn_supported(x, self.n1):
                # graph capture: bias-free convolutions, biases folded into the fused kernels
                h = F.conv2d(_norm_act(self.n1, x), self.c1.weight, None, 1, 1)
                h = F.conv2d(_norm_act(self.n2, h, _fgn_f32(self.c1, "bias")), self.c2.weight, None, 1, 1)
                s = x if isinstance(self.skip, nn.Identity) else F.conv2d(x, self.skip.weight, None)
                if _fgn_nhwc16(h, s):
                    return _fgn_add_bias(h, s, _fgn_bsum(self))
                return h + _fgn_bsum(self).to(h.dtype)[None, :, None, None] + s
            # native path; inside graph capture _norm_act routes silu(group_norm(.)) through the fused kernel
            h = self.c1(_norm_act(self.n1, x))
            h = self.c2(_norm_act(self.n2, h))
            return h + self.skip(x)
        h = self.n1(x)
        if self.n_modalities and mod is not None:
            h = self._mod(self.f1, h, mod)
        h = self.c1(F.silu(h))
        h = self.n2(h)
        if self.n_modalities and mod is not None:
            h = self._mod(self.f2, h, mod)
        h = self.c2(F.silu(h))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, width=64, mults=(1, 2, 4), blocks_per_level=2,
                 t_in=T_IN, t_out=T_OUT, channels=C, n_modalities=0, cond="norm",
                 anchor="last"):
        super().__init__()
        # What the predicted delta is added to. `last` is the shipped behaviour and
        # the default, so every checkpoint packed before 2026-08-18 -- which is all
        # of them, including 887158 -- loads and runs exactly as before. `blend`
        # learns a per-output-frame mix of the last observed frame and the input
        # window's time-mean: t+1 is nearly persistence, t+20 nearly climatology,
        # and 20 scalars let the network say where the crossover is.
        self.anchor = anchor
        self.t_in, self.t_out, self.c = t_in, t_out, channels
        self.n_modalities = n_modalities
        self.cond = cond if n_modalities else "none"
        nb = n_modalities if self.cond == "norm" else 0
        cin, cout = t_in * channels, t_out * channels
        chans = [width * m for m in mults]

        self.stem = nn.Conv2d(cin, chans[0], 3, padding=1)
        self.downs, self.pools = nn.ModuleList(), nn.ModuleList()
        prev = chans[0]
        for i, ch in enumerate(chans):
            self.downs.append(nn.ModuleList(
                [Block(prev if b == 0 else ch, ch, n_modalities=nb)
                 for b in range(blocks_per_level)]))
            prev = ch
            self.pools.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                              if i < len(chans) - 1 else nn.Identity())
        self.mid1 = Block(prev, prev, n_modalities=nb)
        self.mid2 = Block(prev, prev, n_modalities=nb)
        self.ups, self.upsamples = nn.ModuleList(), nn.ModuleList()
        for i, ch in reversed(list(enumerate(chans))):
            self.upsamples.append(nn.ConvTranspose2d(prev, ch, 4, stride=2, padding=1)
                                  if i < len(chans) - 1 else nn.Identity())
            prev = ch if i < len(chans) - 1 else prev
            self.ups.append(nn.ModuleList(
                [Block(prev + (ch if b == 0 else 0), ch, n_modalities=nb)
                 for b in range(blocks_per_level)]))
            prev = ch
        self.out_norm = nn.GroupNorm(8, prev)
        self.out = nn.Conv2d(prev, cout, 3, padding=1)
        if anchor == "blend":
            self.anchor_logit = nn.Parameter(torch.zeros(t_out))
        if self.cond == "enc":
            self.mod_emb = nn.Embedding(n_modalities, width * 4)
            self.film = nn.ModuleList([nn.Linear(width * 4, 2 * ch) for ch in chans])

    def forward(self, x):
        # Always the real row: the simulated one exists only for training.
        idx = (torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
               if self.n_modalities else None)
        m = idx if self.cond == "norm" else None
        e = self.mod_emb(idx) if self.cond == "enc" else None
        h = self.stem(x)
        skips = []
        for i, (level, pool) in enumerate(zip(self.downs, self.pools)):
            for blk in level:
                h = blk(h, m)
            if e is not None:
                g, b = self.film[i](e).chunk(2, dim=-1)
                h = h * (1 + g[:, :, None, None]) + b[:, :, None, None]
            skips.append(h)
            h = pool(h)
        h = self.mid2(self.mid1(h, m), m)
        for level, up, skip in zip(self.ups, self.upsamples, reversed(skips)):
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            for blk in level:
                h = blk(h, m)
        if _FUSED[0] and self.anchor != "blend" and _fgn_nhwc16(h, x):
            y = F.conv2d(_norm_act(self.out_norm, h), self.out.weight, None, 1, 1)
            if _fgn_nhwc16(y):
                return _fgn_add_last_bias(y, _fgn_f32(self.out, "bias"), x, (self.t_in - 1) * self.c, self.c)
            y = y + self.out.bias[None, :, None, None]
        else:
            y = self.out(_norm_act(self.out_norm, h))
        last = x[:, (self.t_in - 1) * self.c: self.t_in * self.c]
        if self.anchor != "blend":
            return y + last.repeat(1, self.t_out, 1, 1)
        b, _, h, w = x.shape
        wmean = x[:, :self.t_in * self.c].reshape(
            b, self.t_in, self.c, h, w).mean(dim=1)
        a = torch.sigmoid(self.anchor_logit).to(x.dtype)[None, :, None, None, None]
        base = (a * last[:, None] + (1 - a) * wmean[:, None])
        return y + base.reshape(b, self.t_out * self.c, h, w)


SIGMA_GLOBAL = 0.0563870259


class UncertaintyHead(nn.Module):
    def __init__(self, t=T_IN, c=C, width=48, mults=(1, 2, 4), extra_channels=0):
        super().__init__()
        self.t, self.c = t, c
        self.extra_channels = extra_channels
        cin = t * c * 2 + 1 + extra_channels
        chans = [width * m for m in mults]
        self.stem = nn.Conv2d(cin, chans[0], 3, padding=1)
        self.downs, self.pools = nn.ModuleList(), nn.ModuleList()
        prev = chans[0]
        for i, ch in enumerate(chans):
            self.downs.append(Block(prev, ch))
            prev = ch
            self.pools.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                              if i < len(chans) - 1 else nn.Identity())
        self.mid = Block(prev, prev)
        self.ups, self.upsamples = nn.ModuleList(), nn.ModuleList()
        for i, ch in reversed(list(enumerate(chans))):
            self.upsamples.append(nn.ConvTranspose2d(prev, ch, 4, stride=2, padding=1)
                                  if i < len(chans) - 1 else nn.Identity())
            prev = ch if i < len(chans) - 1 else prev
            self.ups.append(Block(prev + ch, ch))
            prev = ch
        self.out_norm = nn.GroupNorm(8, prev)
        self.out = nn.Conv2d(prev, 2 * t * c, 3, padding=1)

    def forward(self, window, pred, log_scale, extra=None):
        b, _, h, w = window.shape
        plane = log_scale.reshape(b, 1, 1, 1).expand(b, 1, h, w)
        parts = [window, pred, plane] + ([extra] if self.extra_channels else [])
        d = self.widths(torch.cat(parts, dim=1))
        n = self.t * self.c
        return d[:, :n], d[:, n:]          # widths below and above the prediction

    def widths(self, zin):
        """The width field (B, 2 * T * C, H, W) from the concatenated input (window, prediction, log scale, spread)."""
        z = self.stem(zin)
        skips = []
        for blk, pool in zip(self.downs, self.pools):
            z = blk(z)
            skips.append(z)
            z = pool(z)
        z = self.mid(z)
        for blk, up, skip in zip(self.ups, self.upsamples, reversed(skips)):
            z = up(z)
            z = blk(torch.cat([z, skip], dim=1))
        if _FUSED[0] and _fgn_nhwc16(z):
            d = F.conv2d(_norm_act(self.out_norm, z), self.out.weight, None, 1, 1)
            d = (_fgn_clamp_exp_bias(d, _fgn_f32(self.out, "bias"), -9.0, -2.0) if _fgn_nhwc16(d) else
                 torch.exp((d + self.out.bias[None, :, None, None]).clamp(-9.0, -2.0)))
        else:
            d = torch.exp(self.out(_norm_act(self.out_norm, z)).clamp(-9.0, -2.0))
        return d


_state = {}


def _load():
    if not _state:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(_CKPT, map_location="cpu")
        # The weights are stored fp16 (four fp32 width-96 members would not fit the 256 MB cap) and on a GPU they
        # stay fp16: all GPU arithmetic is fp16, so the stored tensors are exactly what executes, and the fields are
        # O(1) after per-window normalisation. Off CUDA the weights are widened to fp32, where fp16 would be slower.
        dt = torch.float16 if device == "cuda" else torch.float32

        def f32(sd):
            return {k: (v.to(dt) if v.is_floating_point() else v) for k, v in sd.items()}
        # channels_last on the GPU removes the layout conversions cuDNN otherwise inserts to reach its NHWC
        # kernels (60-66 of ~300 kernels per member); outputs move by one to two fp16 ulps. Off CUDA everything
        # stays fp32 and contiguous.
        fmt = torch.channels_last if device == "cuda" else torch.contiguous_format
        models = []
        for sd in ck["models"]:
            # mults/blocks default to the shipped architecture when absent, so a
            # checkpoint packed before 2026-08-17 still loads unchanged.
            m = UNet(width=ck["width"], n_modalities=2 if ck["film"] else 0,
                     cond=ck.get("cond", "enc"),
                     mults=tuple(int(v) for v in str(ck.get("mults") or "1,2,4").split(",")),
                     blocks_per_level=int(ck.get("blocks") or 2),
                     anchor=ck.get("anchor") or "last")
            m.load_state_dict(f32(sd))
            models.append(m.to(device=device, dtype=dt, memory_format=fmt).eval())
        head = UncertaintyHead(width=ck["head_width"],
                               extra_channels=80 if ck.get("head_spread") else 0)
        head.load_state_dict(f32(ck["head"]))
        head = head.to(dtype=dt, memory_format=fmt)
        _state.update(models=models, head=head.to(device).eval(), device=device)
        if device == "cuda":
            try:
                _fgn_load(_HERE)
                _state["fused"] = True
            except Exception as e:  # noqa: BLE001 -- fall back to native GroupNorm in the graphs
                _state["fused"] = False
                _log(f"fused GN unavailable: {e!r}"[:100])
    return _state["models"], _state["head"], _state["device"]


_SETUP_DONE = []


def _setup():
    """Load, and capture every batch bucket, once, outside any timed region. Idempotent.

    Doing this at *import* rather than in ``SubmissionModel.__init__`` is deliberate.
    ``competition_info/submission.md`` says the ingestion program "also supports" the
    class, without saying which entry point wins when a file offers both -- and this
    repository's own ``tools/run_and_score.py`` resolves that ambiguity by preferring
    the module-level ``predict``. If the real ingestion does the same, work parked in
    ``__init__`` would never run. Import happens before either entry point is called
    under both interfaces, so this is the one place that cannot be bypassed.

    Wrapped, because a submission that raises at import scores zero on every subscore: any failure here leaves
    predict() to load lazily and, if graphs are unavailable, to run the eager path.
    """
    if _SETUP_DONE:
        return
    _SETUP_DONE.append(True)
    try:
        _load()
        # Every bucket is captured here, at import (predict() calls _ensure_graphs too, a no-op once captured).
        # Capturing lazily on the first call was measured slower on the timed call that follows it: the GPU is
        # then still at its idle clock (765 vs ~1400 MHz on an A100), and import-time capture keeps it boosted.
        # Each bucket's self-check runs the eager path at its shape, so no separate eager warm-up is needed.
        for b in sorted(set(_GRAPH_PAR) | set(_GRAPH_SEQ)):
            _ensure_graphs(b)
    except Exception:      # noqa: BLE001 -- never let setup break inference
        pass


def _sanitise(uv):
    """Non-finite u, v are read as 0 -- how the release encodes cells without a measurement -- so a corrupted input
    cell costs its own cell, not the whole window (a NaN would otherwise poison the window's RMS scale and zero its
    output). The identity on finite inputs."""
    return torch.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0)


def _window_scale(uv):
    """RMS speed over the input window's valid cells, per window, on device.

    Cells inside the airfoil body and outside the PIV field of view are stored as
    exact zeros. They contribute nothing to a sum of squares, so only the count of
    valid cells needs the mask -- and it must be the window-local mask, since a
    single window is all we ever see.
    """
    tot = uv.pow(2).sum(dim=(1, 2, 3, 4))
    nz = (uv.abs() >= EPS).any(dim=-1).any(dim=1)
    return (tot / (T_IN * nz.sum(dim=(1, 2)).clamp_min(1))).sqrt() + EPS


def predict(input_array, metadata=None):
    """(N, T_in, H, W, C) -> dict with prediction (N, T_out, H, W, C) plus bounds."""
    _d0 = _diag(nvml=False) if _NCALL[0] < 3 else None
    t0 = time.perf_counter()
    x = np.asarray(input_array, dtype=np.float32)
    if x.ndim == 5 and x.shape[1:] == (T_IN, 32, 64, 3) and len(x):
        _ensure_graphs(len(x))
    out, path = None, "eager"
    if _GRAPHS and x.ndim == 5 and x.shape[1:] == (T_IN, 32, 64, 3):
        try:
            out, path = _predict_graphed(x), "graph"
        except Exception:  # noqa: BLE001 -- fall through to the eager path
            out = None
    if out is None:
        out = _predict_eager(x)
    _NCALL[0] += 1
    if _NCALL[0] <= 3:
        _d1 = _diag(nvml=_NCALL[0] == 1)    # the SM clock once, after the untimed first call, before the timed one
        _log(f"call{_NCALL[0]} {tuple(x.shape)} {path} {1e3 * (time.perf_counter() - t0):.1f}ms "
             f"sm={_d1[1]} dflt={_d1[2] - _d0[2]} t={_d0[0]:.2f}")
    return out


def _predict_eager(x):
    """The eager path (torch's native ops): used when graphs are unavailable, and as the reference each captured
    bucket is checked against at import."""
    n, _, h, w, _ = x.shape
    models, head, device = _load()

    # Assembled in fp16 on the GPU (the precision the forecast is computed in) and widened to float32 at return.
    out_dt = torch.float16 if device == "cuda" else torch.float32
    pred = torch.empty((n, T_OUT, h, w, 3), dtype=out_dt, device=device)
    lo = torch.empty_like(pred)
    hi = torch.empty_like(pred)

    # Half precision on the GPU (see _load); fp32 off CUDA.
    with torch.no_grad():
        for i in range(0, n, _BATCH):
            xb = torch.from_numpy(x[i:i + _BATCH]).to(device, non_blocking=True)
            uv = _sanitise(xb[..., :2])
            scale = _window_scale(uv)
            s = scale[:, None, None, None, None]
            p0 = next(models[0].parameters())
            fmt = (torch.channels_last if p0.is_contiguous(memory_format=torch.channels_last)
                   and p0.dim() == 4 else torch.contiguous_format)
            xin = (uv / s).permute(0, 1, 4, 2, 3).reshape(-1, T_IN * C, h, w)
            xin = xin.to(p0.dtype).contiguous(memory_format=fmt)

            # Back to fp32 for the statistics: the spread is a difference of nearby
            # numbers and the interval widths are compared against sigma_global in
            # absolute terms, neither of which wants half precision.
            outs = torch.stack([m(xin) for m in models]).float()
            y = outs.mean(0)
            extra = None
            if head.extra_channels:
                # De-normalized to physical units and divided by sigma_global, so
                # the spread reaches the head on the same footing as the widths it
                # emits. Both the ratio and its log go in: it spans two orders of
                # magnitude between quiescent cells and the wake.
                r = outs.std(0, unbiased=True) * scale[:, None, None, None] / SIGMA_GLOBAL
                extra = torch.cat([r, torch.log(r.clamp_min(1e-4))], dim=1)
            hdt = next(head.parameters()).dtype
            below, above = head(xin, y.to(hdt), torch.log(scale).to(hdt),
                                None if extra is None else extra.to(hdt))
            below, above = below.float(), above.float()
            y_plain = y
            if GAIN_DOSE > 0:
                y = _matched_gain(outs, y)

            yy = y.reshape(-1, T_OUT, C, h, w).permute(0, 1, 3, 4, 2) * s
            yp = y_plain.reshape(-1, T_OUT, C, h, w).permute(0, 1, 3, 4, 2) * s
            bb = below.reshape(-1, T_OUT, C, h, w).permute(0, 1, 3, 4, 2)
            aa = above.reshape(-1, T_OUT, C, h, w).permute(0, 1, 3, 4, 2)
            sl = slice(i, i + _BATCH)
            pred[sl, ..., :2] = yy.to(out_dt)
            pred[sl, ..., 2] = 0.0                       # p is unmeasured
            # Interval widths are absolute, as sigma_global is, so they are not
            # rescaled by the window scale.
            lo[sl, ..., :2] = (yp - bb).to(out_dt)      # the intervals bracket the plain mean
            lo[sl, ..., 2] = 0.0
            hi[sl, ..., :2] = (yp + aa).to(out_dt)
            hi[sl, ..., 2] = 0.0

    # A non-finite prediction, or a non-finite or reversed bound, zeroes every
    # subscore, so guarantee it here rather than trusting the forward pass.
    for t in (pred, lo, hi):
        torch.nan_to_num_(t, nan=0.0, posinf=0.0, neginf=0.0)
    torch.minimum(lo, pred, out=lo)
    torch.maximum(hi, pred, out=hi)

    return {
        # The fp16 values widened exactly: the harness receives contiguous float32 arrays, the FAQ's cheapest form.
        "prediction": pred.float().cpu().numpy(),
        "lower": lo.float().cpu().numpy(),
        "upper": hi.float().cpu().numpy(),
    }

#: Batch shapes captured as CUDA graphs at setup. Up to eight windows the four members replay concurrently on their
#: own streams, then one post graph (mean, head, gain, assembly); from sixteen up one sequential graph is faster.
#: A call of up to eight windows replays its exact bucket; larger calls are decomposed greedily into 64 / 32 / 16
#: and one small bucket (see _pieces).
_GRAPH_PAR = (1, 2, 3, 4, 5, 6, 7, 8)
_GRAPH_SEQ = (16, 32, 64)
_GRAPHS = {}


def _pre(xb, models):
    """Per-window normalisation and channel packing, exactly as the eager path."""
    n, _, h, w, _ = xb.shape
    uv = _sanitise(xb[..., :2])
    scale = _window_scale(uv)
    s = scale[:, None, None, None, None]
    xin = (uv / s).permute(0, 1, 4, 2, 3).reshape(-1, T_IN * C, h, w)
    p0 = next(models[0].parameters())
    fmt = torch.channels_last if p0.is_contiguous(memory_format=torch.channels_last) and p0.dim() == 4 else torch.contiguous_format
    return xin.to(p0.dtype).contiguous(memory_format=fmt), scale, s, h, w, n


def _member0(m, xb, models):
    """Member 0's forward, also returning the normalised input and the window scale it computed (for the post graph)."""
    xin, scale = _pre(xb, models)[:2]
    return m(xin), xin, scale.float().contiguous()


def _post_fused(xb, mout, models, head, pre):
    """The first post graph of a parallel bucket with the glue kernels: the head's input and the prediction plane.
    Returns (out32, z): z is the head's input when only the prediction plane of out32 is written (the bounds follow in
    _post_bounds), None when out32 is complete (the native computation). Together with _post_bounds the same
    computation as torch.stack(_post(...)).float() to within one fp16 ulp on a few elements (fp32 accumulation order).
    Used only while capturing fused graphs; pre = (xin, scale) from member 0's graph."""
    xin, scale = pre
    if not (head.extra_channels == 80 and len(mout) == 4 and _fgn_nhwc16(xin, *mout) and xin.shape[1] == 40):
        return torch.stack(_post(xb, torch.stack(mout), models, head)).float(), None
    return _fgn_assemble(mout, None, scale, part=0), _fgn_head_input(mout, xin, scale)


def _post_bounds(z, o, mout, scale, head):
    """The second post graph: the interval head on its prepared input, then the bounds planes of o."""
    d = head.widths(z)
    if not _fgn_nhwc16(d):
        d = d.to(torch.float16).contiguous(memory_format=torch.channels_last)
    if not _fgn_nhwc16(d):
        raise RuntimeError("head widths are not fp16 channels_last")
    return _fgn_assemble(mout, d, scale, part=1, o=o)


def _post(xb, outs, models, head):
    """Mean, interval head and output assembly, exactly as the eager path."""
    xin, scale, s, h, w, n = _pre(xb, models)
    outs = outs.float()
    y = outs.mean(0)
    extra = None
    if head.extra_channels:
        r = outs.std(0, unbiased=True) * scale[:, None, None, None] / SIGMA_GLOBAL
        extra = torch.cat([r, torch.log(r.clamp_min(1e-4))], dim=1)
    hdt = next(head.parameters()).dtype
    below, above = head(xin, y.to(hdt), torch.log(scale).to(hdt),
                        None if extra is None else extra.to(hdt))
    below, above = below.float(), above.float()
    y_plain = y
    if GAIN_DOSE > 0:
        y = _matched_gain(outs, y)
    yy = y.reshape(-1, T_OUT, C, h, w).permute(0, 1, 3, 4, 2) * s
    yp = y_plain.reshape(-1, T_OUT, C, h, w).permute(0, 1, 3, 4, 2) * s
    bb = below.reshape(-1, T_OUT, C, h, w).permute(0, 1, 3, 4, 2)
    aa = above.reshape(-1, T_OUT, C, h, w).permute(0, 1, 3, 4, 2)
    pred = torch.zeros((n, T_OUT, h, w, 3), dtype=torch.float16, device=xb.device)
    lo = torch.zeros_like(pred)
    hi = torch.zeros_like(pred)
    pred[..., :2] = yy.to(pred.dtype)
    lo[..., :2] = (yp - bb).to(pred.dtype)
    hi[..., :2] = (yp + aa).to(pred.dtype)
    for t in (pred, lo, hi):
        torch.nan_to_num_(t, nan=0.0, posinf=0.0, neginf=0.0)
    return pred, torch.minimum(lo, pred), torch.maximum(hi, pred)


def _capture(fn, *args):
    """Warm once on a side stream, then capture. Returns (graph, static outputs)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(*args)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn(*args)
    return g, out


class _Graphed:
    """One batch shape, captured. Each member graph owns its memory pool, so the four can replay concurrently
    without sharing intermediate buffers; the post graph reads their static outputs after the streams are joined."""

    def __init__(self, n, models, head, device, parallel):
        self.n, self.parallel = n, parallel
        self.static = torch.zeros((n, T_IN, 32, 64, 3), device=device)
        if parallel:
            self.streams = [torch.cuda.Stream() for _ in models]
            self.gm, mout, pre = [], [], None
            for i, m in enumerate(models):
                if i == 0 and _FUSED[0]:
                    g, (o, *pre) = _capture(lambda xb, m=m: _member0(m, xb, models), self.static)
                else:
                    g, o = _capture(lambda xb, m=m: m(_pre(xb, models)[0]), self.static)
                self.gm.append(g)
                mout.append(o)
            # The fp16 -> fp32 widening and the stacking of prediction / lower / upper happen inside the graph, so a
            # call ends in device-to-host copies of one float32 buffer (0.55 vs ~1.0 ms for three casts and three
            # copies at N = 5 on an A100). Values are the same fp16 numbers widened exactly. With the fused kernels the
            # post graph is two graphs: the first writes the prediction plane (it needs only the members), whose copy
            # to the host then overlaps the second (the interval head and the bounds; see run).
            self.gp2, self.cs = None, torch.cuda.Stream()
            if _FUSED[0]:
                self.gp, (self.out32, z) = _capture(lambda xb: _post_fused(xb, mout, models, head, pre), self.static)
                if z is not None:
                    self.gp2, _ = _capture(lambda xb: _post_bounds(z, self.out32, mout, pre[1], head), self.static)
            else:
                self.gp, self.out32 = _capture(
                    lambda xb: torch.stack(_post(xb, torch.stack(mout), models, head)).float(), self.static)
            # Page-locked landing buffers, each handed out ONCE and never reused, so an
            # array returned by one call can never be overwritten by a later one. They are
            # faulted in here, at setup, so a fresh process's first calls do not pay the page faults of new host
            # allocations. Four cover the harness's warm-up + timed pair with margin; later calls fall back to one
            # pageable copy.
            try:
                self.pinned = [torch.empty(self.out32.shape, dtype=torch.float32, pin_memory=True)
                               for _ in range(4)]
            except Exception:  # noqa: BLE001 -- no page-locked memory: keep the graph, copy pageable
                self.pinned = []
        else:
            self.g, self.out = _capture(
                lambda xb: _post(xb, torch.stack([m(_pre(xb, models)[0]) for m in models]),
                                 models, head), self.static)

    def run(self, x_np, host=None):
        """Replay for x_np (k <= n windows). Parallel buckets return the (3, k, ...) float32 device buffer, or, with a
        page-locked `host` buffer of its full shape (k == n), copy into it and return it."""
        k = x_np.shape[0]
        self.static[:k].copy_(torch.from_numpy(np.ascontiguousarray(x_np)))
        if k < self.n:      # pad by repeating the last window; the rows are sliced off
            self.static[k:].copy_(self.static[k - 1:k].expand(self.n - k, -1, -1, -1, -1))
        if self.parallel:
            cur = torch.cuda.current_stream()
            for g, s in zip(self.gm, self.streams):
                s.wait_stream(cur)
                with torch.cuda.stream(s):
                    g.replay()
            for s in self.streams:
                cur.wait_stream(s)
            self.gp.replay()
            if self.gp2 is not None:
                if host is not None:
                    # The prediction plane is final: its copy overlaps the head (the copy engine beside the SMs).
                    self.cs.wait_stream(cur)
                    with torch.cuda.stream(self.cs):
                        host[0].copy_(self.out32[0], non_blocking=True)
                self.gp2.replay()
                if host is not None:
                    host[1:].copy_(self.out32[1:])      # synchronous on this stream
                    self.cs.synchronize()
                    return host
            elif host is not None:
                host.copy_(self.out32)                  # synchronous: pinned, non_blocking=False
                return host
        else:
            self.g.replay()
            return [o[:k] for o in self.out]
        return self.out32[:, :k]


_GRAPH_FAILED = False


def _required_buckets(n):
    """Same decomposition as _pieces, using the complete supported bucket set."""
    needed = set()
    while n > 0:
        if n > 8:
            b = next((b for b in reversed(_GRAPH_SEQ) if b <= n), 8)
        else:
            b = next(b for b in _GRAPH_PAR if b >= n)
        needed.add(b)
        n -= min(b, n)
    return needed


#: Fused graphs differ from the native eager path by fp16 rounding of intermediate activations: on this
#: synthetic check the measured max |diff| is one fp16 ulp (2.4e-4 .. 2.7e-4 at every bucket, H100 and A100),
#: on 292 held-out windows at most 9.8e-4; a wrong kernel differs at O(0.1-1). Same bar as the native graphs.
_FUSED_TOL = 2e-3


def _check_graph(graph, x, ref, b, tol):
    runs = [graph.run(x)]
    if getattr(graph, "pinned", None):
        # The harness's path: straight into a page-locked landing buffer (overwritten by the call that later gets it).
        runs.append(graph.run(x, graph.pinned[-1]))
    for got in runs:
        for key, out in zip(("prediction", "lower", "upper"), got):
            d = np.abs(out.float().cpu().numpy() - ref[key]).max()
            if not d < tol:
                raise RuntimeError(f"graph self-check failed at {b}: {key} {d}")


def _ensure_graphs(n):
    """Prepare by shape on the first call; no input values or predictions persist.

    The official synthetic warm-up has the scored batch's shape. This path is
    equally valid on a first real call; it simply charges setup to that call.
    """
    global _GRAPH_FAILED
    models, head, device = _load()
    if device != "cuda" or _GRAPH_FAILED:
        return
    needed = _required_buckets(n) - _GRAPHS.keys()
    if not needed:
        return
    try:
        with torch.no_grad():
            for b in sorted(needed):
                # Validate every new bucket on synthetic data, never hidden values.
                x = np.random.default_rng(b).normal(
                    0.15, 0.05, size=(b, T_IN, 32, 64, 3)).astype(np.float32)
                x[..., 2] = 0.0
                ref = _predict_eager(x)
                graph = None
                if _state.get("fused"):
                    try:
                        _FUSED[0] = True
                        try:
                            graph = _Graphed(b, models, head, device, parallel=b in _GRAPH_PAR)
                        finally:
                            _FUSED[0] = False
                        _check_graph(graph, x, ref, b, _FUSED_TOL)
                        graph.fused = True
                    except Exception as e:  # noqa: BLE001 -- recapture with torch's native ops
                        _state["fused"] = False
                        _log(f"fused GN disabled at bucket {b}: {e!r}"[:100])
                        graph = None
                        torch.cuda.synchronize()
                if graph is None:
                    graph = _Graphed(b, models, head, device, parallel=b in _GRAPH_PAR)
                    _check_graph(graph, x, ref, b, 2e-3)
                _GRAPHS[b] = graph
            torch.cuda.synchronize()
    except Exception:
        _GRAPHS.clear()
        _GRAPH_FAILED = True


def _pieces(n):
    """Split a call of n windows into (bucket, count) replays.

    Up to eight windows: the exact bucket (all of 1..8 are captured). Above eight: an exact-sum greedy
    decomposition (64s, 32s, 16s, then one small bucket), because padding a compute-bound batch is real work --
    measured, padding 33 -> 64 ran 0.6x eager, while 32 + 1 is neutral.
    """
    small = sorted(b for b in _GRAPHS if b <= 8)
    large = sorted((b for b in _GRAPHS if b > 8), reverse=True)
    out, k = [], n
    while k > 0:
        if k > 8:
            # `large` may lack a bucket <= k (9..15 windows, or a bucket that failed to capture); the biggest small
            # bucket then covers eight, exactly as _required_buckets planned.
            b = next((b for b in large if b <= k), None)
            if b is None:            # 9..15: the biggest small bucket, then the rest
                b = small[-1]
            out.append((b, min(b, k)))
        else:
            b = next((b for b in small if b >= k), None)
            if b is None:            # only if no small bucket covers k (never, in practice)
                b = large[-1]
            out.append((b, min(b, k)))
        k -= out[-1][1]
    return out


def _predict_graphed(x):
    pieces = _pieces(x.shape[0])
    if len(pieces) == 1 and _GRAPHS[pieces[0][0]].parallel:
        # The common case (the harness sends five windows per call): one replay, one copy.
        b, k = pieces[0]
        g = _GRAPHS[b]
        with torch.no_grad():
            if g.pinned and k == b:
                host = g.run(x, g.pinned.pop())
            else:
                host = g.run(x).contiguous().cpu()     # padded or pinned buffers spent
        a = host.numpy()
        return {"prediction": a[0], "lower": a[1], "upper": a[2]}
    parts = {"prediction": [], "lower": [], "upper": []}
    i = 0
    with torch.no_grad():
        for b, k in pieces:
            out = _GRAPHS[b].run(x[i:i + k])
            if _GRAPHS[b].parallel:
                out = out.unbind(0)
            for key, o in zip(parts, out):
                parts[key].append(o.float().cpu().numpy())      # float32 payload, fp16 values
            i += k
    return {k: (v[0] if len(v) == 1 else np.concatenate(v)) for k, v in parts.items()}


_setup()
try:
    _log(f"import {time.perf_counter() - _T_IMPORT:.1f}s graphs={len(_GRAPHS)} fused={_state.get('fused')}/{_FGN.get('origin')} fb={sum(getattr(g, 'fused', False) for g in _GRAPHS.values())} t={_T_EPOCH:.2f}/{_diag()[0]:.2f} sm={_diag()[1]} "
         f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
except Exception:  # noqa: BLE001
    pass


class SubmissionModel:
    """The class interface of submission.md. Setup already ran at import (_setup is idempotent), so construction
    and load_checkpoint do nothing further; both entry points compute exactly the same thing."""

    def __init__(self, submission_dir=None, device="cpu"):
        # The checkpoint path is resolved relative to this file, so the argument is
        # accepted and ignored rather than trusted; _CKPT is what the archive ships.
        self._dir = submission_dir
        _setup()          # idempotent, and normally already done at import

    def load_checkpoint(self, path=None, device=None):
        """Accepted because submission.md says the ingestion calls it when
        ``model.pth`` exists. ``_load()`` is idempotent, so this is a no-op after
        ``__init__`` and a load if the ingestion calls it first."""
        _load()

    def predict(self, input_array, metadata=None):
        return predict(input_array, metadata)
