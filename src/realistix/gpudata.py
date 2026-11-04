"""GPU-resident window sampling.

The DataLoader path was the bottleneck, not the GPU. Windows were sliced, scaled
and collated in the main Python process, so training was CPU-bound: three runs on
three otherwise-idle GPUs of a 192-core machine starved each other, and runs
placed on cluster nodes whose CPUs were busy with other users' work fell to
0.17 it/s against 16 it/s on a quiet host. That also explains why ``pq`` looked
wrong -- it schedules on free GPU memory, which is the right signal for a
GPU-bound job and the wrong one for this.

The original justification for avoiding workers was that the submission rules
require ``num_workers=0`` -- but that rule governs the evaluation container, and
the submission does not use a DataLoader at all (``predict`` receives a numpy
array). Training was never constrained by it.

At the evaluation resolution the whole corpus is ~4 GB, so it lives in GPU memory
and batches are gathered there: no workers, no host round-trip, no collation.
"""

from __future__ import annotations

import torch

from .data import T_IN, T_OUT, frame_range, load_traj
from .normalize import EPS


class GPUWindows:
    """All trajectories of one split resident on the GPU; batches gathered by index."""

    def __init__(self, trajs, split, device="cuda", tag=0, frames="all",
                 t_extra: int = 0, phases: int = 1, sampling: str = "window",
                 dilate_prob: float = 0.0, dilate_factors=(2.0,), dilate_seed: int = 0):
        # Time dilation (2026-09-10 screen, the owner's idea): with probability
        # `dilate_prob` a sampled window is re-read from the trajectory at a different
        # clock -- factor 2 takes every second raw frame (a double-speed window, raw
        # span 80), factor 0.5 linearly interpolates midpoints (half speed, raw span
        # 20). Input and target are dilated alike and the per-window RMS scale is
        # recomputed on the dilated input, exactly as the submission would see it.
        # Decisions draw from a private generator so the (trajectory, start) sequence
        # is identical to an undilated control's. Zero = the shipped sampler, untouched.
        self.dilate_prob = float(dilate_prob)
        self.dilate_factors = tuple(float(f) for f in dilate_factors)
        for f in self.dilate_factors:
            if f not in (2.0, 0.5):
                raise ValueError("dilate factors supported: 2 and 0.5")
        self.dilate_gen = None
        # Realised augmentation statistics (no RNG calls): drawn rows, rows dilated per
        # factor, rows that fell back to native because the trajectory was too short,
        # and per-trajectory counts of dilated draws. Read by the trainer at each eval.
        self.dilate_stats = dict(drawn=0, fallback=0, **{f"factor_{f:g}": 0 for f in self.dilate_factors})
        self.dilate_traj_counts = None
        if self.dilate_prob > 0:
            self.dilate_gen = torch.Generator(device=device)
            self.dilate_gen.manual_seed(int(dilate_seed) + 7919)
        # ``phases`` > 1 loads every trajectory once per decimation phase of the
        # raw 64x128 grid (2026-08-31): the same band-limited field under an
        # independent PIV-noise draw. Uniform sampling over the expanded index is
        # uniform over (window) x (phase), so the marginal over windows -- and
        # therefore the real:sim mix, which reads ``n_mix`` -- is unchanged.
        # ``t_extra`` extends every sampled window by that many frames beyond the
        # target, for unrolled training (2026-08-22): the extra frames are the
        # target of a second, autoregressive application of the model. They are
        # normalized by the same window scale and stashed on ``_last_y2`` rather
        # than widening the sample tuple, mirroring ``_last_regime``.
        self.t_extra = t_extra
        if sampling not in ("window", "trajectory"):
            raise ValueError(f"unknown sampling unit: {sampling}")
        self.sampling = sampling
        self.fields, starts, owner = [], [], []
        # The regime each trajectory belongs to, carried alongside the windows.
        # Conditioning on it is what makes a simulated trajectory and its real
        # partner share a pathway: their descriptor is identical by construction,
        # and only the modality token differs.
        pairs = [(t, ph) for t in trajs for ph in range(phases)]
        self.regime = torch.tensor([[float(t.re), float(t.aoa)] for t, _ in pairs],
                                   dtype=torch.float32, device=device)
        for i, (t, ph) in enumerate(pairs):
            f = load_traj(t.path(split), phase=ph)             # (T, 2, H, W)
            self.fields.append(torch.from_numpy(f).to(device))
            # Under the in-grid holdout the sampler must never draw a window whose
            # target reaches into the evaluation tail; `frame_range` leaves a full
            # window of gap between the two.
            lo, hi = frame_range(f.shape[0], frames)
            n = hi - lo - (T_IN + T_OUT + t_extra)
            starts.append(torch.arange(n + 1, device=device) + lo)
            owner.append(torch.full((n + 1,), i, device=device, dtype=torch.long))
        self.starts = torch.cat(starts)
        self.owner = torch.cat(owner)
        self.counts = torch.tensor([s.numel() for s in starts], device=device)
        self.offsets = self.counts.cumsum(0) - self.counts
        self.n_mix = self.starts.numel() // phases   # unique windows, for the mix
        self.device = device
        self.tag = tag
        self.win = torch.arange(T_IN + T_OUT + t_extra, device=device)

    def __len__(self):
        return int(self.starts.numel())

    #: Optional per-window sampling weights (2026-09-11 frontier-weighting screen).
    #: None = the shipped uniform draw, bit for bit. Set through set_traj_weights().
    window_weights = None

    def set_traj_weights(self, per_traj):
        """Weight each trajectory's windows by ``per_traj[i]`` (relative; all-ones or
        None restores the shipped uniform sampler). Only the plain window sampler
        without an ``angles`` restriction supports weights."""
        if per_traj is None:
            self.window_weights = None
            return
        w = torch.as_tensor(per_traj, dtype=torch.float64, device=self.device)
        if w.numel() != len(self.fields) or (w < 0).any() or w.sum() <= 0:
            raise ValueError("one non-negative weight per trajectory, not all zero")
        if torch.all(w == w[0]):
            self.window_weights = None          # uniform: keep the shipped RNG path
            return
        if self.sampling != "window":
            raise ValueError("trajectory weights need --sampling window")
        self.window_weights = w[self.owner]     # per window; trajectories differ in length

    def sample(self, batch: int, generator=None, angles=None):
        # ``angles`` (2026-09-05, cross-angle meta-update screen): restrict the draw to
        # windows whose trajectory has one of these angles of attack. None = all.
        if self.window_weights is not None and angles is None and self.sampling == "window":
            idx = torch.multinomial(self.window_weights, batch, replacement=True, generator=generator)
        elif self.window_weights is not None:
            raise ValueError("trajectory weights do not compose with an angles restriction")
        elif self.sampling == "trajectory":
            pool = torch.arange(len(self.fields), device=self.device)
            if angles is not None:
                keep = sum(self.regime[:, 1] == float(a) for a in angles) > 0
                pool = pool[keep]
                if pool.numel() == 0:
                    raise ValueError(f"no trajectories at angles {list(angles)}")
            o = pool[torch.randint(pool.numel(), (batch,), device=self.device,
                                   generator=generator)]
            u = torch.rand(batch, device=self.device, generator=generator)
            idx = self.offsets[o] + (u * self.counts[o]).long()
        elif angles is None:
            idx = torch.randint(len(self), (batch,), device=self.device, generator=generator)
        else:
            aoa = self.regime[self.owner][:, 1]
            pool = torch.nonzero(sum((aoa == float(a)) for a in angles) > 0).squeeze(1)
            if pool.numel() == 0:
                raise ValueError(f"no windows at angles {list(angles)}")
            idx = pool[torch.randint(pool.numel(), (batch,), device=self.device, generator=generator)]
        o, s = self.owner[idx], self.starts[idx]
        self._last_regime = self.regime[o]
        h, w = self.fields[0].shape[-2:]
        out = torch.empty(batch, T_IN + T_OUT + self.t_extra, 2, h, w,
                          device=self.device)
        if self.dilate_prob > 0 and not self.t_extra:
            rows_a, rows_b = self._dilated_rows(o, s, batch)
            for j in torch.unique(o):
                m = o == j
                f = self.fields[int(j)]
                out[m] = 0.5 * (f[rows_a[m]] + f[rows_b[m]])
            return self._finish(out)
        # Trajectories differ in length, so gather once per distinct trajectory.
        for j in torch.unique(o):
            m = o == j
            rows = s[m][:, None] + self.win[None, :]
            out[m] = self.fields[int(j)][rows]
        return self._finish(out)

    def _dilated_rows(self, o, s, batch):
        """Raw-frame index pairs (B, T) whose mean is the (possibly dilated) window.

        Undilated and double-speed rows have rows_a == rows_b (the mean is exact);
        half-speed rows pair floor/ceil raw frames so odd output frames are midpoints.
        A double-speed window needs 2T-1 raw frames, so its start is redrawn
        uniformly over [first start, last admissible start - (T - 1)] of the same
        trajectory (its last raw index is then exactly the last native window's), so
        a dilated window never reaches past what an undilated one may see, and no
        start is over-represented by clamping. Trajectories with fewer than 2T - 1
        usable frames stay native.
        """
        T = self.win.numel()
        rows_a = s[:, None] + self.win[None, :]
        rows_b = rows_a.clone()
        u = torch.rand(batch, device=self.device, generator=self.dilate_gen)
        pick = torch.randint(len(self.dilate_factors), (batch,), device=self.device,
                             generator=self.dilate_gen)
        sel = u < self.dilate_prob
        self.dilate_stats["drawn"] += int(batch)
        if self.dilate_traj_counts is None:
            self.dilate_traj_counts = torch.zeros(len(self.fields), dtype=torch.long, device=self.device)
        for k, f in enumerate(self.dilate_factors):
            m = sel & (pick == k)
            if not bool(m.any()):
                continue
            if f == 2.0:
                first_start = self.starts[self.offsets[o[m]]]
                n_legal = self.counts[o[m]] - (T - 1)          # s + 2T - 2 <= last native end
                ok = n_legal >= 1                              # trajectory long enough
                self.dilate_stats["fallback"] += int((~ok).sum())
                self.dilate_stats[f"factor_{f:g}"] += int(ok.sum())
                self.dilate_traj_counts += torch.bincount(o[m][ok], minlength=len(self.fields))
                u2 = torch.rand(int(m.sum()), device=self.device, generator=self.dilate_gen)
                s2 = first_start + (u2 * n_legal.clamp_min(1).float()).long()
                r = torch.where(ok[:, None], s2[:, None] + 2 * self.win[None, :],
                                s[m][:, None] + self.win[None, :])
                rows_a[m] = r
                rows_b[m] = r
            else:
                self.dilate_stats[f"factor_{f:g}"] += int(m.sum())
                self.dilate_traj_counts += torch.bincount(o[m], minlength=len(self.fields))
                half = self.win // 2
                rows_a[m] = s[m][:, None] + half[None, :]
                rows_b[m] = s[m][:, None] + ((self.win + 1) // 2)[None, :]
        return rows_a, rows_b

    def _finish(self, w):
        x, y = w[:, :T_IN], w[:, T_IN:T_IN + T_OUT]
        # Same definition as the CPU path: RMS speed over the input window's valid
        # cells, with validity computed window-locally (a trajectory-wide mask would
        # train on a scale the submission cannot reproduce).
        tot = x.pow(2).sum(dim=(1, 2, 3, 4))
        nz = (x.abs() >= EPS).any(dim=2).any(dim=1)
        scale = (tot / (T_IN * nz.sum(dim=(1, 2)).clamp_min(1))).sqrt() + EPS
        s = scale[:, None, None, None, None]
        b = x.shape[0]
        if self.t_extra:
            y2 = w[:, T_IN + T_OUT:]
            self._last_y2 = (y2 / s).reshape(b, -1, *x.shape[-2:])
        else:
            self._last_y2 = None
        return ((x / s).reshape(b, -1, *x.shape[-2:]),
                (y / s).reshape(b, -1, *y.shape[-2:]),
                scale,
                torch.full((b,), self.tag, device=self.device, dtype=torch.long),
                self._last_regime)


class MixedGPUWindows:
    """Real and simulated windows on the GPU, sampled in proportion to their counts."""

    def __init__(self, real_trajs, sim_trajs, device="cuda", frames="all",
                 t_extra: int = 0, phases: int = 1, sim_batch_frac: float | None = None,
                 sampling: str = "window", dilate_prob: float = 0.0,
                 dilate_factors=(2.0,), dilate_domain: str = "real", dilate_seed: int = 0):
        if dilate_domain not in ("real", "all"):
            raise ValueError("dilate_domain must be 'real' or 'all'")
        d_real = dict(dilate_prob=dilate_prob, dilate_factors=dilate_factors,
                      dilate_seed=dilate_seed)
        d_sim = d_real if dilate_domain == "all" else {}
        # ``phases`` applies to the *real* trajectories only: simulated fields are
        # noise-free CFD, so their decimation phases are redundant to 0.2%.
        # `frames` applies to the *real* trajectories only. The in-grid holdout
        # exists to keep evaluation windows out of training, and evaluation is on
        # real measurements; simulated trajectories are never scored, so withholding
        # their tails would only throw away supervision.
        if sim_batch_frac is not None and not 0 <= sim_batch_frac <= 1:
            raise ValueError("sim_batch_frac must be in [0, 1]")
        if sim_batch_frac is not None and not (real_trajs and sim_trajs):
            raise ValueError("sim_batch_frac requires both real and simulated trajectories")
        self.sim_batch_frac = sim_batch_frac
        self.real = (GPUWindows(real_trajs, "real", device, tag=0, frames=frames,
                                t_extra=t_extra, phases=phases, sampling=sampling, **d_real)
                     if real_trajs else None)
        self.sim = (GPUWindows(sim_trajs, "sim", device, tag=1, t_extra=t_extra,
                               sampling=sampling, **d_sim)
                    if sim_trajs else None)
        self._last_y2 = None

    def last_y2(self):
        """The normalized frames beyond the target of the last sample, or None."""
        if self.real is None:
            return self.sim._last_y2
        if self.sim is None:
            return self.real._last_y2
        return self._last_y2

    def __len__(self):
        return (len(self.real) if self.real else 0) + (len(self.sim) if self.sim else 0)

    def sample(self, batch: int, generator=None, angles=None):
        self._last_y2 = None
        if self.real is None:
            return self.sim.sample(batch, generator, angles)
        if self.sim is None:
            return self.real.sample(batch, generator, angles)
        frac = (self.sim.n_mix / (self.sim.n_mix + self.real.n_mix)
                if self.sim_batch_frac is None else self.sim_batch_frac)
        n_sim = int(round(batch * frac))
        if n_sim == 0 or n_sim == batch:
            ds = self.sim if n_sim else self.real
            result = ds.sample(batch, generator, angles)
            self._last_y2 = ds._last_y2
            return result
        a = self.real.sample(batch - n_sim, generator, angles)
        b = self.sim.sample(n_sim, generator, angles)
        if self.real._last_y2 is not None:
            self._last_y2 = torch.cat([self.real._last_y2, self.sim._last_y2])
        else:
            self._last_y2 = None
        return tuple(torch.cat([u, v]) for u, v in zip(a, b))
