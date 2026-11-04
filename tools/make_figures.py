#!/usr/bin/env python3
"""Generate the figures and animations used by README.md.

    python tools/make_figures.py --out docs

Everything spatial is computed live from the shipped checkpoint and the released
data, so the figures can be regenerated after any change rather than drifting out of
date. Summary series that came from the leaderboard or from finished sweeps are
inlined here with their source noted.

The field panels render velocity **magnitude** with streamlines over it and the NACA
4418 section drawn explicitly -- see `realistix.viz` for the geometry, which is
fitted from the data's own always-zero mask rather than assumed. Magnitude rather
than the streamwise component because the streamwise field carries a large free-stream
mean that flattens the wake; magnitude plus streamlines shows the separated region and
the shed vortices, which is what the model is actually being asked to forecast.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "starting_kit"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from realistix.data import build, load_traj  # noqa: E402
from realistix.models.uncertainty import UncertaintyHead  # noqa: E402
from realistix.splits import REAL_DIR, SIM_DIR  # noqa: E402
from realistix import viz  # noqa: E402
from posthoc import collect  # noqa: E402
from train_final_head import Ensemble  # noqa: E402

RUNS = "runs"
# matplotlib's default property cycle: C0 blue for the series, C1 orange for
# the reverted submissions, C2 green for the best one.
ACC, WARN, OK = "C0", "C1", "C2"


def _cbar(fig, im, ax, label="Velocity Magnitude"):
    # `aspect` is length/width and defaults to 20, so narrowing the bar also caps
    # how tall it can be -- the forecast figure's colorbars ended up spanning one row
    # of three. Raise it so a thin bar can still run the full height of what it
    # labels.
    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.008, aspect=42)
    cb.set_label(label, color=viz.FG, fontsize=8)
    cb.ax.tick_params(colors=viz.MUTED, labelsize=7)
    cb.outline.set_edgecolor(viz.MUTED)
    return cb


def _banner(ax, text):
    """A panel label. Plain type -- these used to be letter-spaced bold small-caps
    (`" ".join(text)` inserting a space between every character), which is a poster
    affectation that costs vertical room and reads worse at README width."""
    # A real title, not ax.text at an axes-fraction offset. constrained_layout
    # cannot see free-floating text, so it laid the suptitle out as though the panel
    # labels did not exist and the two collided the moment the figure got tight.
    # set_title participates in the layout, which fixes the overlap and removes the
    # guessed title band in one move.
    ax.set_title(text, fontsize=8.5, color=viz.FG, pad=2.5)


# The honest proxy, not the shipped build. `hd_clean` + `clean_s*` is what ships, and
# it trains on every regime -- so scoring it on `val_aoa` is in-sample, and a figure
# captioned "held-out angle" would be showing the model answering windows it trained
# on. `px2_*` is the same recipe with AoA=20 and the two val_re Reynolds numbers
# genuinely withheld (docs/archive/HANDOFF_2026-09-10.md section 3), which is what the caption claims.
PROXY_HEAD = f"{RUNS}/px2_head/best.pt"


def forecast_gif(out, head_ck=PROXY_HEAD, n=3, aoa=20):
    """Measurement, forecast and error over the 20 forecast frames, held-out angle.

    ``head_ck`` must be a run that held AoA=20 out. Pass the shipped head only to
    inspect the artefact; anything it produces on ``val_aoa`` is in-sample fit and
    must not be shown as a forecast of unseen data.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(head_ck, map_location=dev)
    head = UncertaintyHead(width=ck["args"]["width"], extra_channels=80).to(dev)
    head.load_state_dict(ck["head"]); head.eval()
    ens = Ensemble(ck["paths"], dev)
    d = collect(ens, head, DataLoader(build("real", "val_aoa", stride=40), batch_size=64), dev)
    p, t = d["pred"], d["tgt"]
    # Pick the windows with the most temporal variation -- a still wake shows nothing.
    var = viz.speed(t[..., :2], axis=-1).std(axis=(1, 2, 3))
    idx = np.argsort(var)[-n:][::-1]

    # Height derived, not guessed: three columns of 2:1 panels share whatever width
    # is left after the two colorbars, and the row height follows from that. 1.75*n
    # left roughly 0.7in of vertical slack that showed up as gaps between the rows.
    _w, _cb = 10.0, 1.7
    fig, axes = plt.subplots(n, 3, constrained_layout=True,
                             figsize=(_w, (_w - _cb) / 3 / 2 * n + 0.42))
    axes = np.atleast_2d(axes)
    # One scale for every row, so the three windows are comparable rather than each
    # normalised to itself.
    sp_all = [(viz.speed(t[w, ..., :2], axis=-1), viz.speed(p[w, ..., :2], axis=-1))
              for w in idx]
    hi = max(viz.vmax_for(a) for a, _ in sp_all)
    ehi = float(np.percentile(np.concatenate([np.abs(b - a).ravel() for a, b in sp_all]), 99.5))
    ims = []
    for r, (w, (sp_t, sp_p)) in enumerate(zip(idx, sp_all)):
        err = np.abs(sp_p - sp_t)
        msk = (t[w, 0, ..., 0] == 0) & (t[w, 0, ..., 1] == 0)
        for c, (title, u, v, mag, kw) in enumerate((
                ("measured", t[w, ..., 0], t[w, ..., 1], sp_t, dict(vmin=0, vmax=hi)),
                ("forecast", p[w, ..., 0], p[w, ..., 1], sp_p, dict(vmin=0, vmax=hi)),
                # Error goes white-to-red rather than inferno. Inferno's low end is black,
                # so on a white page a near-perfect forecast rendered as a solid dark
                # rectangle -- the panel was loudest exactly where it had least to say.
                # With `Reds`, zero error recedes into the page and the eye lands on
                # the wake, which is where the error actually lives.
                ("|error|", None, None, err, dict(vmin=0, vmax=ehi, cmap="Reds")))):
            ax = axes[r, c]
            if u is None:
                im = viz.field_panel(ax, mag[0] * 0, mag[0] * 0, aoa_deg=aoa,
                                     streamlines=False, mask=msk, **kw)
                im.set_data(np.where(msk | viz.inside_airfoil(aoa), np.nan, mag[0]))
            else:
                im = viz.field_panel(ax, u[0], v[0], aoa_deg=aoa, mask=msk,
                                     streamlines=False, **kw)
            if r == 0:
                _banner(ax, title)
            ims.append([im, mag, msk, ax,
                        (u, v) if u is not None else None, None])
    _cbar(fig, ims[0][0], list(axes[:, :2].ravel()))
    _cbar(fig, ims[2][0], list(axes[:, 2]), "|error|")
    fig.suptitle(f"NACA 4418 · {aoa}° angle of attack, held out of training entirely"
                 f" · 0.4 s forecast", color=viz.FG, fontsize=10)

    hidden = viz.inside_airfoil(aoa)

    def upd(f):
        for rec in ims:
            im, mag, msk, ax, uv, old = rec
            im.set_data(np.where(msk | hidden, np.nan, mag[f]))
            if uv is not None:
                if old is not None:
                    viz.clear_stream(old)
                rec[5] = viz.stream(ax, uv[0][f], uv[1][f], msk | hidden)
        return [rec[0] for rec in ims]

    animation.FuncAnimation(fig, upd, frames=t.shape[1], interval=140, blit=False).save(
        os.path.join(out, "forecast.gif"), writer=animation.PillowWriter(fps=7), dpi=82)
    plt.close(fig)
    return float(np.abs(p[..., :2] - t[..., :2]).mean())


def sim_vs_real_gif(out, stem="13950_20", frames=40):
    """The domain gap at one regime, after the per-window normalisation."""
    aoa = int(stem.split("_")[1])

    def norm(path):
        f = load_traj(path)[:frames]
        s = np.sqrt((f ** 2).sum() / max((np.abs(f) > 1e-12).sum(), 1))
        return f / s

    r, s = norm(os.path.join(REAL_DIR, stem + ".h5")), norm(os.path.join(SIM_DIR, stem + ".h5"))
    # Sized from the panels rather than guessed. Each is 64x32 at equal aspect, so a
    # panel of height h occupies 2h of width and any figure wider than
    # 2*(2h) + colorbar just opens a gap down the middle -- which is what the old
    # 10.2x2.5 did, and what a first attempt at 9.6x2.15 did worse by shrinking the
    # height instead of the width. h = 2.0in of panel plus ~0.35in of title.
    fig, axes = plt.subplots(1, 2, figsize=(8.9, 2.35), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.0)
    hi = viz.vmax_for(viz.speed(r, axis=1).mean(0))
    ims = []
    for ax, arr, ttl in ((axes[0], r, "measured · PIV"), (axes[1], s, "simulated · CFD")):
        # no_signal detects the PIV laser shadow and must not be applied to the CFD
        # panel: there it flags the genuine low-speed core of the separated region as
        # "no data", which is exactly wrong -- the CFD has no shadow at all, and that
        # absence is part of what this comparison is showing.
        msk = (arr[0, 0] == 0) & (arr[0, 1] == 0)
        if ttl.startswith("measured"):
            msk = msk | viz.no_signal(arr)
        im = viz.field_panel(ax, arr[0, 0], arr[0, 1], aoa_deg=aoa, vmin=0,
                             vmax=hi, mask=msk, streamlines=False)
        _banner(ax, ttl)
        ims.append([im, arr, msk, ax, None])
    _cbar(fig, ims[-1][0], axes[1])
    fig.suptitle(f"NACA 4418 · Re {stem.split('_')[0]}, {aoa}° · "
                 f"same regime, same normalisation", color=viz.FG, fontsize=10)
    hidden = viz.inside_airfoil(aoa)

    def upd(f):
        for rec in ims:
            im, arr, msk, ax, old = rec
            im.set_data(np.where(msk | hidden, np.nan, viz.speed(arr[f])))
            if old is not None:
                viz.clear_stream(old)
            rec[4] = viz.stream(ax, arr[f, 0], arr[f, 1], msk | hidden)
        return [rec[0] for rec in ims]

    animation.FuncAnimation(fig, upd, frames=len(r), interval=110, blit=False).save(
        os.path.join(out, "sim_vs_real.gif"), writer=animation.PillowWriter(fps=9), dpi=88)
    plt.close(fig)


def regimes_png(out, aoas=(0, 5, 10, 15, 20), re=13950):
    """The five angles side by side -- the grid the private set draws novel angles from."""
    # Five panels is width-bound, not height-bound: the panel height follows from
    # whatever width is left after the colorbar, so the height has to be derived or
    # the figure letterboxes. h = (W - colorbar) / n / 2, plus the title band.
    _w = 2.72 * len(aoas)
    fig, axes = plt.subplots(1, len(aoas),
                             figsize=(_w, (_w - 0.75) / len(aoas) / 2 + 0.55),
                             constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.0)
    hi = None
    for ax, aoa in zip(np.atleast_1d(axes), aoas):
        f = load_traj(os.path.join(REAL_DIR, f"{re}_{aoa}.h5"))
        sp = viz.speed(f, axis=1).mean(0)
        msk = (f == 0).all(axis=(0, 1)) | viz.no_signal(f)
        if hi is None:
            hi = viz.vmax_for(sp, msk)
        im = viz.field_panel(ax, f[:, 0].mean(0), f[:, 1].mean(0), aoa_deg=aoa,
                            vmin=0, vmax=hi, mask=msk, density=0.9)
        _banner(ax, f"{aoa}°")
    _cbar(fig, im, np.atleast_1d(axes)[-1])
    fig.suptitle(f"NACA 4418 · Re {re} · time-mean speed at each measured angle",
                 color=viz.FG, fontsize=10)
    fig.savefig(os.path.join(out, "regimes.png"), dpi=150)
    plt.close(fig)


def progression(out):
    """Leaderboard history. Scores are as returned by Codabench."""
    subs = [("FNO baseline\n+ constant band", 77.780), ("U-Net w64\ncomposite loss", 78.414),
            ("3x w64 ensemble\n+ learned intervals", 79.964), ("w96 + EMA\nfp16", 80.095),
            ("all trajectories\n+ spread head", 80.792), ("domain-specific\nnorm", 80.746),
            ("folded + fp16", 80.889), ("2 members\n(a mistake)", 80.677),
            ("loss re-derived\nfrom the weights", 81.157),
            ("inference re-timed\n(reverted)", 81.137), ("29% fewer op sites\n(reverted)", 81.144)]
    fig, ax = plt.subplots(figsize=(9.4, 3.4), constrained_layout=True)
    y = [s for _, s in subs]
    ax.plot(range(len(y)), y, "-o", color=ACC, lw=1.7, ms=5, zorder=3)
    for i in (7, 9, 10):
        ax.plot([i], [y[i]], "o", color=WARN, ms=7, zorder=4)
    ax.plot([8], [y[8]], "o", color=OK, ms=9, zorder=5)
    for i, (lab, s) in enumerate(subs):
        ax.annotate(lab, (i, s), textcoords="offset points", xytext=(0, -32 if i % 2 else 13),
                    ha="center", fontsize=6.6, color=viz.MUTED)
    ax.set_xticks([]); ax.set_ylabel("final score")
    ax.margins(x=0.06, y=0.26)
    ax.set_title("leaderboard progression — best 81.157; the three orange points were reverted",
                 fontsize=10, color=viz.FG)
    ax.grid(axis="y", zorder=0)
    fig.savefig(os.path.join(out, "progression.png"), dpi=150)
    plt.close(fig)


def rel_sweep(out):
    """The relative-L2 weight sweep (tools/weighted.py, 3-6 seeds, both held-out splits)."""
    w = [0.5, 1, 2, 4, 8, 16]
    aoa = [0, 0.205, 0.323, 0.464, 0.613, 0.621]
    re = [0, 0.183, 0.355, 0.481, 0.533, 0.443]
    fig, ax = plt.subplots(figsize=(5.6, 3.2), constrained_layout=True)
    ax.semilogx(w, aoa, "-o", color=ACC, label="held-out angle", base=2)
    ax.semilogx(w, re, "-s", color=WARN, label="held-out Reynolds", base=2)
    ax.axvline(8, color=viz.MUTED, ls="--", lw=1)
    ax.set_xlabel("weight on the relative-L2 loss term (scorer weights it 0.524)")
    ax.set_ylabel("gain, weighted score")
    ax.set_title("the loss had this at 0.5, against a plain MSE at 1.0",
                 fontsize=10, color=viz.FG)
    ax.legend(frameon=False, labelcolor=viz.FG); ax.grid()
    fig.savefig(os.path.join(out, "rel_sweep.png"), dpi=150)
    plt.close(fig)


def spectrum(out):
    """Where the sim-to-real translator's energy sits, by wavenumber (held-out regimes)."""
    bands = ["k<2", "2-4", "4-8", "8-16", "k>16"]
    series = {"simulated": [1.11, 1.19, 2.48, 3.16, 1.62],
              "transported": [0.71, 0.51, 0.67, 0.83, 1.02],
              "+ broadband noise": [0.65, 0.51, 0.80, 1.76, 6.22]}
    fig, ax = plt.subplots(figsize=(6.0, 3.2), constrained_layout=True)
    x = np.arange(len(bands))
    for (lab, v), c, m in zip(series.items(), ("C7", ACC, WARN), ("o", "s", "^")):
        ax.semilogy(x, v, "-" + m, color=c, label=lab, ms=5)
    ax.axhline(1.0, color=viz.MUTED, lw=1, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(bands)
    # The default log locator labels only 10^0 over this range, which leaves the
    # reader unable to put a number on any point. Label the decade the data spans.
    ax.set_yticks([0.5, 1, 2, 4, 8])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0.4, 8)
    ax.set_xlabel("wavenumber band"); ax.set_ylabel("energy / measured")
    ax.set_title("fluctuation energy of the translated fields", fontsize=10, color=viz.FG)
    ax.legend(frameon=False, fontsize=8, labelcolor=viz.FG); ax.grid()
    fig.savefig(os.path.join(out, "spectrum.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs")
    ap.add_argument("--only", default=None, help="comma-separated subset")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    viz.light_style()
    todo = args.only.split(",") if args.only else [
        "progression", "rel_sweep", "spectrum", "regimes", "sim_vs_real", "forecast"]
    if "progression" in todo: progression(args.out); print("wrote progression.png")
    if "rel_sweep" in todo: rel_sweep(args.out); print("wrote rel_sweep.png")
    if "spectrum" in todo: spectrum(args.out); print("wrote spectrum.png")
    if "regimes" in todo: regimes_png(args.out); print("wrote regimes.png")
    if "sim_vs_real" in todo: sim_vs_real_gif(args.out); print("wrote sim_vs_real.gif")
    if "forecast" in todo:
        mae = forecast_gif(args.out); print(f"wrote forecast.gif (mean |error| {mae:.4f})")


if __name__ == "__main__":
    main()
