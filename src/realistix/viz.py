"""Field rendering: velocity magnitude, streamlines, and the airfoil drawn explicitly.

The grid is 32x64 and **row 0 is the top**: `docs/archive/competition_overview.md` section 3.3
verified against the files that y decreases with row index, and that the laser-shadow
wedge sits in the *lower* left. So the vertical axis is inverted below. This is
not cosmetic -- drawing them the other way up mirrors the flow, puts the shadow above
the model and makes the sections read at the opposite incidence.

One consequence worth stating rather than hiding: in this data the labelled angle
pitches the nose *down*. Pooling the always-zero mask per angle and tracking its ends,
the upstream end moves to higher row index (lower y) as the angle grows while the
downstream end moves to lower row index, and the measured fast flow is above the model
with the shadow below it. The figures therefore show the chord rising from leading to
trailing edge. That is what the release contains; no sign convention is asserted here
beyond it.

The section is the competition's stated **NACA4418**, and the geometry is measured from
this release rather than borrowed.

**The obvious external source does not apply.** RealPDEBench (arXiv 2601.01829) documents
a Foil scenario that looks like ours -- circulating water tunnel, PIV plus immersed-boundary
CFD, angles 0-20 -- but as a *tapered NACA0025*, root chord 100 mm, tip 20 mm, measurement
plane at z = 50 mm. It is **not the same dataset**:

| | this release | RealPDEBench foil |
|---|---|---|
| real / sim trajectories | 82 / 100 | 99 / 99 |
| raw grid | 64 x 128 | 128 x 256 |
| dt | 0.02 s | 2.5e-3 s |
| duration | 12-17 s | 9.97 s |
| measured Re | 3750-26761 (sim to 27975) | 2968-17031 |
| nominal Re step | 1275 | ~781 |

Grid and dt could be subsampling. A *longer recording* and a 57% *higher* Reynolds number
could not be. So this is an extended or separate campaign on the same rig, the NACA0025
label is not established for it, and the competition's own NACA4418 stands.

The data agrees, as far as it can. In the immersed-boundary CFD a cell inside the body has
velocity identically constant for the whole run, so zero temporal variance isolates the
solid footprint exactly. Fitting sections to that footprint at chord 17.53 cells:

| section | solid cells contained | excess cells |
|---|---|---|
| **NACA4418** | **100% at all five angles** | **32** |
| NACA0018 | 99% | 37 |
| NACA0025 | 97% | 106 |
| NACA0030 | 99% | 154 |

4418 is the only one that contains every solid cell at every angle, and it does so with the
tightest excess. Consistent with the footprint's centreline, which arches by about a cell
between 10% and 60% of the chord.

**Do not over-read that.** At 32x64 the two candidate thicknesses differ by ~1 cell and 4%
camber by 0.7 of a cell, against a footprint 3 cells thick. The grid cannot settle the
profile; it can only say that 4418 is not contradicted and fits best.

Size and position, from the same footprint:

* chord **17.53 cells = 60.0 mm** -- the smallest section containing every solid cell;
* leading edge at x = 0.60, chord line at y = 15.30, rotation about 0.60 of the chord;
* 100% containment at 0, 5, 10, 15 and 20 degrees.

**The leading edge sits essentially on the inflow boundary**, half a cell in, so the frame
carries no approach flow at all. That is most of why the nose looks clipped and why none of
these panels resemble a textbook aerofoil rendering.

**Five attempts got here, and the failures are the useful part.** Chord 14 from the extent
of the scorer's always-zero mask; 20 from requiring the section to *cover* that mask, which
is "outside the field of view **or** inside the body" -- body union shadow, so it bounds
neither side; 14.6 from reading the benchmark's "z = 50 mm" as a chord; 17.5 with a
NACA0025 borrowed from a dataset that turned out to be a different one. And the rotation
centre is *not* well determined: it lands anywhere from grid x = 7.5 to 11.1 depending on
the section assumed, so an earlier claim in this file that every fit put it at 7.0-7.7 was
wrong.

*The Reynolds route deserves its own warning.* Inverting `U * c / nu` against the measured
`re` gives 20.09 cells with 2.2% spread over 81 trajectories and `corr(Re, U) = 0.9992`.
Consistent and wrong: it needs the organisers' reference velocity and the water
temperature, neither published. At 60 mm it implies nu = 9.0e-7, water at 24-25 C, which is
plausible and not assumable. **The tight spread measures how cleanly U scales with Re, not
whether the constant relating them is the one assumed.** Precision is not accuracy.

In chords the domain is **3.7 c x 1.8 c** with the leading edge *on* the inflow
boundary, so the frame is a wake measurement with essentially no approach flow -- most
of why it looks unlike a textbook aerofoil rendering.
"""

from __future__ import annotations

import json
import os

import numpy as np

GEOM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "airfoil_geom.json")

# Dark ground, hot flow, pale body -- and a background that the low-magnitude end of
# the colormap sits close to, so quiescent fluid recedes and the wake carries the eye.
# Light theme. The figures used to render on a dark panel, which looked good in
# isolation and wrong everywhere README.md is actually read -- GitHub in light mode,
# a PDF, a printed page. These are matplotlib's own defaults where matplotlib has a
# sensible one.
BG = "white"
FG = "black"
MUTED = "#4d4d4d"
# Cells the PIV never measured. This used to be BG, i.e. the panel colour, which
# only reads as "no data" while the panel is dark -- on white it would vanish into
# the margin and README.md promises a grey. It is its own constant now because it is
# data, not theme.
MASK = "#bdbdbd"
# Streamlines sit over the colormap, never over the page, so this is tuned for
# contrast against turbo's bright mid-range rather than against the background.
# Lightening it for the light theme was a mistake: it made them fainter.
STREAM = "#c8ccd0"
BODY = "#f2f2f0"
BODY_EDGE = "#2b2d30"
# Velocity magnitude keeps turbo and the error panel keeps inferno. Both are field
# renderings rather than theme, and a perceptually-ordered warm map is the
# convention this literature reads in; neither depends on the page being dark.
CMAP = "turbo"


def naca4(code: str = "4418", n: int = 160):
    """Upper and lower surfaces of a NACA 4-digit section, in chord coordinates.

    Returns ``(x, y)`` tracing the closed contour from the trailing edge over the
    upper surface to the leading edge and back along the lower surface. ``x`` runs
    0 (leading edge) to 1 (trailing edge); ``y`` is perpendicular, in chord units.
    """
    m = int(code[0]) / 100.0            # maximum camber
    p = int(code[1]) / 10.0             # its chordwise position
    t = int(code[2:]) / 100.0           # maximum thickness
    # Cosine spacing, so the curvature at the nose is resolved rather than chorded.
    x = (1 - np.cos(np.linspace(0, np.pi, n))) / 2
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                  + 0.2843 * x ** 3 - 0.1036 * x ** 4)      # -0.1036 closes the TE
    if m == 0.0:
        # Symmetric section: no camber line, and the 4-digit formula divides by p^2
        # which is zero for a NACA 00xx. This mattered once NACA0025 became a
        # candidate for the release's section.
        yc = np.zeros_like(x)
        th = np.zeros_like(x)
    else:
        fore = x < p
        yc = np.where(fore, m / p ** 2 * (2 * p * x - x ** 2),
                      m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
        dyc = np.where(fore, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
        th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    return (np.concatenate([xu[::-1], xl[1:]]),
            np.concatenate([yu[::-1], yl[1:]]))


def geometry():
    return json.load(open(GEOM))


def airfoil_polygon(aoa_deg: float, g: dict | None = None):
    """The section in grid coordinates at one angle of attack: ``(x, y)`` arrays.

    Direction ``(cos a, -sin a)`` in (column, row): the upstream end sits at higher row
    index as the angle grows, which is what the pooled mask shows. Rotation is about
    the quarter chord.
    """
    g = g or geometry()
    xc, yc = naca4(g.get("naca", "4418"))
    a = np.deg2rad(aoa_deg)
    d = np.array([np.cos(a), -np.sin(a)])                 # leading edge -> trailing edge
    nrm = np.array([d[1], -d[0]])                         # normal; camber toward the
                                                          # measured (upper) side
    pivot = np.array([g["x_le"] + g["pivot_frac"] * g["chord"], g["y_chord"]])
    s = (xc - g["pivot_frac"]) * g["chord"]
    n = yc * g["chord"]
    pts = pivot[None, :] + s[:, None] * d[None, :] + n[:, None] * nrm[None, :]
    return pts[:, 0], pts[:, 1]


def inside_airfoil(aoa_deg: float, shape=(32, 64), g: dict | None = None):
    """Boolean (H, W) mask of cells inside the section, by winding on the polygon."""
    from matplotlib.path import Path
    px, py = airfoil_polygon(aoa_deg, g)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.stack([xx.ravel(), yy.ravel()], 1)
    return Path(np.stack([px, py], 1)).contains_points(pts).reshape(h, w)


def draw_airfoil(ax, aoa_deg: float, g: dict | None = None, lw: float = 0.9, z: int = 6):
    px, py = airfoil_polygon(aoa_deg, g)
    ax.fill(px, py, color=BODY, zorder=z, lw=lw, edgecolor=BODY_EDGE, joinstyle="round")


def speed(a, axis=0):
    """Velocity magnitude, with the two components along ``axis``.

    The axis is explicit on purpose. An earlier version inferred it by looking for a
    length-2 dimension, which silently mis-indexed a ``(T, 2, H, W)`` trajectory --
    ``shape[0]`` is the frame count, so it reduced over width instead and returned a
    plausible-looking array of the wrong shape. Pass ``axis=1`` for a trajectory,
    ``axis=-1`` for the scorer's ``(..., H, W, C)`` layout, ``axis=0`` for one frame.
    """
    a = np.asarray(a)
    u, v = np.take(a, 0, axis=axis), np.take(a, 1, axis=axis)
    return np.sqrt(u ** 2 + v ** 2)


def vmax_for(sp, mask=None, at: float = 0.55, gamma: float = 0.6):
    """Colour limit that puts the *median* measured speed at ``at`` of the range.

    The obvious choice -- a high percentile -- saturates here, because unlike the
    reference CFD renderings the bulk of this field is the fast part and the wake is
    the slow part. Anchoring on the median keeps the free stream mid-scale, so the
    wake reads as cool and any genuine acceleration still has headroom to go hot.

    Gamma-aware: ``field_panel`` renders through ``PowerNorm(gamma)``, so placing the
    median at ``at`` of the *colour* range means placing it at ``at ** (1/gamma)`` of the
    data range. Ignoring that leaves the free stream saturated at the hot end and undoes
    the contrast the power norm was added to buy.
    """
    v = sp if mask is None else sp[~mask]
    return float(np.median(v[v > 0]) / at ** (1.0 / gamma))


def no_signal(traj, frac: float = 0.02):
    """Cells the PIV never measured: the model's optical shadow.

    ``traj`` is one trajectory, ``(T, 2, H, W)``. The scorer excludes only *exactly*
    zero targets, but a further 10-15% of the grid sits below 2% of the median speed
    for the whole recording -- the region the model shadows from the laser sheet.
    Those cells hold ~0% of the field's squared velocity, so they are free accuracy
    for any model and part of why `rel_l2` scores sit near 95. Rendering them as slow
    *flow* would be wrong; they are absence of data.
    """
    sp = speed(traj, axis=1).mean(0)
    med = np.median(sp[sp > 0])
    return sp < frac * med


def stream(ax, u, v, mask, density=1.25, z=4):
    """Streamlines for one frame, returned so an animation can replace them.

    Masked cells are held out of the integration: feeding exact zeros to
    ``streamplot`` drags every nearby line into the body, which reads as a stagnation
    region that is not in the data.
    """
    h, w = u.shape
    uu = np.where(mask, np.nan, u)
    vv = np.where(mask, np.nan, v)
    sl = ax.streamplot(np.arange(w), np.arange(h), uu, vv, color=STREAM,
                       linewidth=0.7, density=density, arrowsize=0.0,
                       integration_direction="both", zorder=z)
    sl.lines.set_alpha(0.8)
    return sl


def clear_stream(sl):
    """Remove a streamplot's artists so the next frame can draw its own."""
    for art in (sl.lines, sl.arrows):
        if art is None:
            continue
        try:
            art.remove()
        except (ValueError, AttributeError, NotImplementedError):
            # Older matplotlib refuses to remove a PatchCollection; hiding it is
            # equivalent for a rendered frame.
            try:
                art.set_visible(False)
            except AttributeError:
                pass


def field_panel(ax, u, v, aoa_deg=None, vmin=None, vmax=None, cmap=CMAP,
                streamlines=True, density=1.25, mask=None, gamma=0.6):
    """One styled panel: magnitude image, streamlines over it, airfoil on top.

    ``mask`` marks cells with no measurement. They are held out of the streamline
    integration -- feeding exact zeros to ``streamplot`` drags every nearby line into
    the body, which reads as a stagnation region that is not in the data.

    **Animated callers must pass ``streamlines=False``** and draw their own with
    ``stream()`` inside the update function. This routine discards the streamplot
    handle, so lines it draws can never be cleared: they stay on top of every later
    frame while each new frame draws underneath, and the animation looks like it has
    a fixed set of streamlines with a moving field behind it. That shipped once.
    """
    sp = speed(np.stack([u, v]))
    h, w = sp.shape
    if mask is None:
        mask = (u == 0) & (v == 0)
    if aoa_deg is not None:
        mask = mask | inside_airfoil(aoa_deg, sp.shape)
    # origin="lower" keeps the data coordinate equal to the row index, so the airfoil
    # polygon and the streamline grid are both in (column, row); the *axis* is then
    # inverted below so row 0 draws at the top, per section 3.3.
    # A power norm rather than a linear one. The wake spans the bottom of the range
    # and a linear scale crushes it into one dark blue -- gamma < 1 expands the low
    # end, which is where the structure right behind the body lives.
    from matplotlib.colors import PowerNorm
    norm = PowerNorm(gamma=gamma, vmin=vmin or 0.0, vmax=vmax) if vmax else None
    im = ax.imshow(np.where(mask, np.nan, sp), origin="lower", cmap=cmap,
                   norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax,
                   interpolation="bilinear",
                   extent=(-0.5, w - 0.5, -0.5, h - 0.5), zorder=1)
    im.cmap.set_bad(MASK)
    if streamlines:
        stream(ax, u, v, mask, density)
    if aoa_deg is not None:
        draw_airfoil(ax, aoa_deg)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)                    # row 0 at the top, per section 3.3
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def light_style():
    """Stock matplotlib, with only the deviations the figures genuinely need.

    Deliberately thin. The previous version set fourteen rcParams to build a dark
    theme; the point of this one is that matplotlib's defaults are already the
    house style of every paper and README these figures sit beside, so the only
    honest overrides are a font size that suits panels this small and an explicit
    white save background. The latter is belt and braces -- `savefig.facecolor`
    defaults to following the figure -- but a figure that saves transparent inherits
    whatever the page is, and half of GitHub reads in dark mode, which is the
    failure this whole change exists to undo.

    `rcdefaults()` first, so calling this after `dark_style()` in the same process
    does not leave a dark rcParam standing.
    """
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    plt.rcParams.update({
        "font.size": 9,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
    })
