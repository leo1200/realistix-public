"""SOAP, and its Muon-orthogonalized variant, for matrix-shaped parameters.

Why this exists (2026-08-31). The optimizer is the one component whose measured
attribution *grew* as the recipe improved — Muon re-measured at +0.070 under the
final recipe against +0.037 under the previous one — and the 2026 comparisons
(arXiv:2607.02499, arXiv:2607.20548) put SOAP at-or-above Muon consistently, with
"SOAP-Muon" strongest in selected settings. Both are one screen away once the
update rule exists; this file is that rule.

SOAP (Vyas et al.): run Adam in Shampoo's eigenbasis. Per matrix G = grad(W)
reshaped (out_channels, rest), maintain EMAs of the two gradient covariances,
refresh their eigenbases every ``precond_freq`` steps, rotate the gradient into
the basis, do Adam there, rotate the update back. SOAP-Muon (arXiv:2607.02499):
the same, then Newton-Schulz-orthogonalize the back-projected update, scaled with
Muon's aspect factor so the step size matches Muon's.

Adaptations for this codebase, stated rather than silent:

* **Dimension cap.** A 384-channel 3x3 conv reshapes to 384x3456, and eigh(3456)
  every few steps across ~10 such tensors is most of a training step. Sides whose
  dimension exceeds ``max_dim`` keep the identity basis (the official
  implementation's max-precond-dim behaviour); the left sides here are never
  above 512, so every tensor is at least one-sided-preconditioned.
* **Refresh every 25 steps** rather than the paper's 10 — the paper reports
  robustness across 10-100, and 25 keeps the overhead near 1.3x a plain step at
  our sizes.
* Momentum (``exp_avg``) is stored in the original space and rotated per step;
  the second moment lives in the rotated space and is carried across basis
  refreshes unchanged — both as in the reference implementation.
* 1-D parameters are the caller's problem (AdamW), exactly as with Muon.
"""

from __future__ import annotations

import torch

from .muon import _newton_schulz


class SOAP(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 3e-4, betas=(0.95, 0.95),
                 shampoo_beta: float = 0.95, eps: float = 1e-8,
                 weight_decay: float = 0.0, precond_freq: int = 25,
                 max_dim: int = 2048, muon: bool = False, ns_steps: int = 5):
        defaults = dict(lr=lr, betas=betas, shampoo_beta=shampoo_beta, eps=eps,
                        weight_decay=weight_decay, precond_freq=precond_freq,
                        max_dim=max_dim, muon=muon, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            sb = group["shampoo_beta"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.reshape(p.shape[0], -1).float()
                m, n = g.shape
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(g)
                    st["exp_avg_sq"] = torch.zeros_like(g)
                    st["L"] = (torch.zeros(m, m, device=g.device)
                               if m <= group["max_dim"] else None)
                    st["R"] = (torch.zeros(n, n, device=g.device)
                               if n <= group["max_dim"] else None)
                    st["QL"] = st["QR"] = None
                st["step"] += 1
                t = st["step"]

                if st["L"] is not None:
                    st["L"].mul_(sb).add_(g @ g.T, alpha=1 - sb)
                if st["R"] is not None:
                    st["R"].mul_(sb).add_(g.T @ g, alpha=1 - sb)
                if t == 1 or t % group["precond_freq"] == 0:
                    if st["L"] is not None:
                        st["QL"] = torch.linalg.eigh(
                            st["L"] / (1 - sb ** t)
                            + 1e-30 * torch.eye(m, device=g.device))[1]
                    if st["R"] is not None:
                        st["QR"] = torch.linalg.eigh(
                            st["R"] / (1 - sb ** t)
                            + 1e-30 * torch.eye(n, device=g.device))[1]

                st["exp_avg"].mul_(b1).add_(g, alpha=1 - b1)

                def rot(x):
                    if st["QL"] is not None:
                        x = st["QL"].T @ x
                    if st["QR"] is not None:
                        x = x @ st["QR"]
                    return x

                def unrot(x):
                    if st["QL"] is not None:
                        x = st["QL"] @ x
                    if st["QR"] is not None:
                        x = x @ st["QR"].T
                    return x

                gr = rot(g)
                mr = rot(st["exp_avg"])
                st["exp_avg_sq"].mul_(b2).add_(gr * gr, alpha=1 - b2)
                vhat = st["exp_avg_sq"] / (1 - b2 ** t)
                mhat = mr / (1 - b1 ** t)
                u = unrot(mhat / (vhat.sqrt() + group["eps"]))
                if group["muon"]:
                    # SOAP-Muon: orthogonalize the preconditioned update, Muon's
                    # scale convention so lr transfers from the Muon recipe.
                    u = _newton_schulz(u, steps=group["ns_steps"])
                    u = u * max(1.0, m / n) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(u.reshape(p.shape).to(p.dtype), alpha=-group["lr"])
        return loss
