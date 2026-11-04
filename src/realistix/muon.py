"""Muon: orthogonalized-momentum updates for matrix-shaped parameters.

The last unswept training-procedure axis (2026-08-22): every run in this project
has used AdamW with default betas, and the 111-trial TPE study varied only
AdamW's own knobs. Muon replaces the elementwise Adam update with the nearest
orthogonal matrix to the momentum (Newton-Schulz iteration), which empirically
buys convnets and transformers real accuracy at the same step budget -- i.e. it
*creates* accuracy rather than re-weighting the objective, which is the one
route this week's measurements left open.

Convolution kernels are treated as (out_channels, everything_else) matrices.
Parameters with fewer than 2 dimensions (biases, norms, embeddings) are not
Muon's business by default and should go to AdamW in a second param group by the
caller -- but ``--muon-1d`` overrides that, because a 1-D parameter reshapes to
(n, 1) and the iteration then returns the unit vector, i.e. sign-free normalized
SGD on that tensor. Whether that helps is an open arm, not a claim.

What is measured about this rule, on the shipped members
(`docs/archive/model_analysis.md` section 3): the accumulated update's stable rank is
**84.3** against AdamW's **6.6**, a 13x gap, while the finished weights differ by
only 1.25x -- so the rank signature lives in the update. The weights end 5.5x
larger in Frobenius norm, which is *not* compensated by the GroupNorm affines
(they grow 1.18x), and the per-block residual branches nevertheless measure 28%
weaker: a higher-rank matrix of larger norm has smaller realized gain on low-rank
activations. ``ScaledMomentum`` below is the control that separates the two.
"""

from __future__ import annotations

import torch


def _newton_schulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    """Approximate msign(G): the nearest semi-orthogonal matrix.

    The quintic coefficients are the standard tuned ones; the iteration runs in
    bfloat16 for speed, which is accurate enough because only the *direction* of
    the update survives.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    if g.size(-2) > g.size(-1):
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        s = x @ x.mT
        y = b * s + c * s @ s
        x = a * x + y @ x
    if g.size(-2) > g.size(-1):
        x = x.mT
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0,
                 adavar: bool = False, adavar_beta: float = 0.95,
                 order: str = "momentum_first", transpose: bool = False):
        # ``adavar`` (2026-09-05, docs/IDEAS_BROAD_2026-09-05.md rank 6): coordinate
        # variance adaptation applied AFTER the orthogonalisation -- V = b V + (1-b) O^2,
        # U = O / sqrt(V_hat), rescaled to ||O||_F per tensor -- so the per-layer step
        # magnitude and the Nesterov/NS machinery are untouched and only the direction
        # within the layer changes. SOAP-Muon preconditions *before* orthogonalising,
        # so its null does not test this order. Off by default: bit-identical to before.
        # ``order`` (2026-09-07 optimizer deep dive): "momentum_first" is the shipped
        # rule, msign(EMA(g)). "polar_first" is EMA(msign(g)): each step's gradient is
        # orthogonalised on its own and the momentum averages the polar factors, so
        # directions that persist across steps keep singular value ~1 while directions
        # that are fresh every step average down to ~sqrt((1-b)/(1+b)) = 0.16 at
        # b = 0.95 -- a noise-aware polarisation, where the shipped order re-amplifies
        # whatever survives the average back to unit gain. Both agents of that round
        # ranked it first and it is **DEAD**: -0.055/-0.094 (mim1, lr 0.02), -0.061 on
        # 0/4 folds at lr 0.04, -0.074 against the --zero-rest baseline. It lowers
        # training loss over 200 steps from a common init and loses every held-out
        # read -- the update noise the shipped order re-amplifies is apparently doing
        # work (member diversity, and the momentum-0.98 ensemble result says the same).
        # ``transpose``: flatten as (shape[1], -1) instead of (shape[0], -1), for
        # ConvTranspose2d weights whose first axis is the INPUT channel. A different
        # metric rather than a bug fix, and **NULL**: -0.011, 1/4 folds.
        if order not in ("momentum_first", "polar_first"):
            raise ValueError(f"order must be momentum_first or polar_first, got {order!r}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay,
                        adavar=adavar, adavar_beta=adavar_beta,
                        order=order, transpose=transpose)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                tr = group.get("transpose", False)

                def _flat(t):
                    return (t.transpose(0, 1).reshape(t.shape[1], -1) if tr
                            else t.reshape(t.shape[0], -1))

                def _unflat(f, like):
                    if tr:
                        return f.reshape(like.shape[1], like.shape[0],
                                         *like.shape[2:]).transpose(0, 1)
                    return f.reshape_as(like)

                def _polar(t):
                    f = _flat(t)
                    o_ = _newton_schulz(f, steps=group["ns_steps"])
                    # Scale so the update RMS matches Adam's unit-ish steps across
                    # aspect ratios (Jordan et al.'s max(1, rows/cols)^0.5 factor).
                    return _unflat(o_, t) * max(1.0, f.shape[0] / f.shape[1]) ** 0.5

                if group.get("order", "momentum_first") == "polar_first":
                    og = _polar(g)
                    buf.lerp_(og, 1 - group["momentum"])
                    o = og.lerp(buf, group["momentum"]) if group["nesterov"] else buf.clone()
                else:
                    buf.lerp_(g, 1 - group["momentum"])
                    u = g.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                    P = group.get("P", None)
                    if P is not None and u.ndim == 4:
                        # Input-covariance metric (2026-09-07, Codex's candidate): right-
                        # multiply the momentum's input-channel axis by P = (C + damp I)^-1,
                        # shared over the kernel offsets, THEN orthogonalise, and rescale
                        # to the Frobenius norm the plain polar update would have had so
                        # the per-layer step size is unchanged. The caller refreshes P.
                        # Measured +0.006 (4/4 folds, consistent in sign, inside the rig's
                        # 0.02-0.03 resolution): too small to fund, not refuted.
                        plain = _polar(u)
                        up = torch.einsum("oihw,ij->ojhw", u.float(), P).to(u.dtype)
                        o = _polar(up)
                        o = o * (plain.norm() / o.norm().clamp_min(1e-12))
                    else:
                        o = _polar(u)
                if group["adavar"]:
                    b2 = group["adavar_beta"]
                    if "v" not in state:
                        state["v"] = torch.zeros_like(o, dtype=torch.float32)
                        state["t"] = 0
                    state["t"] += 1
                    v = state["v"]
                    v.mul_(b2).addcmul_(o.float(), o.float(), value=1 - b2)
                    vhat = v / (1 - b2 ** state["t"])
                    u = o.float() / (vhat.sqrt() + 1e-8)
                    u = u * (o.float().norm() / (u.norm() + 1e-12))
                    o = u.to(o.dtype)
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(o, alpha=-group["lr"])
        return loss


class ScaledMomentum(torch.optim.Optimizer):
    """Muon with the orthogonalization removed and nothing else changed.

    The control the mechanism question needs (`docs/archive/model_analysis.md` section 3.5).
    Muon does two things at once: it makes every layer's update the same *size* --
    an orthogonalized matrix has all its singular values equal to 1, times Jordan
    et al.'s aspect-ratio factor -- and it makes it the same *shape*, full rank with
    a flat spectrum. Only the second is distinctive; per-layer step-size
    normalization is old, and if that is where the +0.074 came from then a much
    cheaper rule buys it.

    So: identical momentum, identical Nesterov, identical weight decay, and the
    update rescaled to exactly the Frobenius norm the orthogonalized update would
    have carried -- ``sqrt(min(rows, cols))`` times the same
    ``max(1, rows/cols)**0.5``. Same magnitude, unmodified spectrum. Anything that
    separates the two arms is the orthogonalization and nothing else.

    **It needs its own learning rate.** Measured over 500 steps from a common
    initialization, the raw momentum here has stable rank **1.5** -- it is nearly
    rank one -- so rescaling it to an orthogonalized matrix's norm puts that entire
    norm into a single direction and the realized relative step comes out 3.3x
    Muon's. Run at Muon's own lr it ends *worse than plain AdamW*, which is a
    statement about the step size and not about the update rule. The matched-step
    setting is about ``lr 0.006``.
    """

    def __init__(self, params, lr: float = 0.006, momentum: float = 0.95,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.lerp_(g, 1 - group["momentum"])
                u = g.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                flat = u.reshape(u.shape[0], -1)
                r, c = flat.shape
                target = min(r, c) ** 0.5 * max(1.0, r / c) ** 0.5
                o = u * (target / flat.norm().clamp_min(1e-20))
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(o, alpha=-group["lr"])
        return loss
