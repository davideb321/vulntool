"""Plot footprints: which patches of the grid are actually sampled forest.

Many census plots are not rectangles. A staircase-shaped plot — a union of
several axis-aligned rectangles covering less area than its bounding box —
is a common real-world case. Treating its bounding box as the plot invents
phantom empty patches, which silently corrupts every zero-fraction with no
error raised.

A footprint can be given as:

* ``(xmin, xmax, ymin, ymax)`` — a plain rectangle (the common case, unchanged);
* a list of such rectangles — their **union**, which need not be convex;
* ``holes`` — rectangles **removed** from the above.
* a boolean mask over the patch grid — an escape hatch for shapes that are not
  rectangle unions. The mask is tied to one grid, so it is only valid at the
  scale it was built for.

Conventions:

* a patch belongs to the plot iff its **centre** lies inside the footprint —
  partially covered patches are all-in or all-out, never fractional;
* membership tests use **half-open** intervals ``[min, max)``, so a point on a
  shared edge belongs to exactly one rectangle;
* a hole wins wherever it overlaps a footprint rectangle.
"""
from __future__ import annotations

from dataclasses import dataclass, replace as _dc_replace
from math import gcd
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Footprint:
    """A plot outline: a bounding box plus an optional shape within it.

    Attributes
    ----------
    bbox : (xmin, xmax, ymin, ymax)
        The enclosing rectangle; the patch grid is always laid over this.
    rects : tuple of (xmin, xmax, ymin, ymax) | None
        Rectangles whose union is the sampled area. ``None`` means the whole
        bbox is sampled.
    mask : (N,) bool array | None
        Explicit per-patch validity, in grid order (x fastest). Only usable at
        the grid size it was built for.
    holes : tuple of (xmin, xmax, ymin, ymax) | None
        Rectangles *removed* from the sampled area. Applied after the union
        above, so a hole wins wherever the two overlap. ``holes`` is the last
        field so that positional ``Footprint(bbox, rects, mask)`` construction
        keeps working.
    """

    bbox: Tuple[float, float, float, float]
    rects: Optional[Tuple[Tuple[float, float, float, float], ...]] = None
    mask: Optional[np.ndarray] = None
    holes: Optional[Tuple[Tuple[float, float, float, float], ...]] = None

    @property
    def is_rectangle(self):
        """True when the sampled area is the whole bounding box."""
        return self.rects is None and self.holes is None and self.mask is None

    @property
    def has_shape(self):
        """True when the sampled area is *not* simply the whole bounding box.

        This — never ``rects is None`` — is the test for "needs exact tiling and
        a lattice-derived scale ladder". A holes-only footprint has no ``rects``
        but is every bit as shaped, and giving it the loose rectangular
        treatment would put grid lines across its hole edges.
        """
        return self.rects is not None or self.holes is not None

    @property
    def outer_rects(self):
        """Rectangles whose union is the sampled area *before* holes are cut."""
        return self.rects if self.rects is not None else (self.bbox,)

    @property
    def lattice_rects(self):
        """Rectangles whose edges the patch grid must land on (rects + holes)."""
        return tuple(self.rects or ()) + tuple(self.holes or ())

    def contains(self, x, y):
        """Vectorised half-open point-in-footprint test (union minus holes).

        The single source of truth for membership: :func:`valid_mask`,
        :func:`inside_footprint` and :meth:`area` all route through it, so the
        hole rule is implemented exactly once.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        inside = np.zeros(np.shape(x), dtype=bool)
        for xmin, xmax, ymin, ymax in self.outer_rects:
            inside |= (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)
        for xmin, xmax, ymin, ymax in (self.holes or ()):
            inside &= ~((x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax))
        return inside

    @property
    def width(self):
        return float(self.bbox[1] - self.bbox[0])

    @property
    def height(self):
        return float(self.bbox[3] - self.bbox[2])

    def area(self):
        """Sampled area: the union of the rectangles, less any holes.

        Note this ignores ``mask`` — a mask is tied to one grid, so its area is
        a property of that grid, not of the footprint.
        """
        if self.is_rectangle:
            return self.width * self.height
        # Rectangles may overlap and holes may straddle them, so measure on the
        # lattice of every edge and decide each strip by its midpoint. Clip to
        # the bbox: ground outside it is not sampled ground either way.
        xlo, xhi, ylo, yhi = self.bbox
        xs = sorted({xlo, xhi} | {v for r in self.lattice_rects for v in r[:2]
                    if xlo < v < xhi})
        ys = sorted({ylo, yhi} | {v for r in self.lattice_rects for v in r[2:]
                    if ylo < v < yhi})
        total = 0.0
        for x0, x1 in zip(xs[:-1], xs[1:]):
            for y0, y1 in zip(ys[:-1], ys[1:]):
                if bool(self.contains(0.5 * (x0 + x1), 0.5 * (y0 + y1))):
                    total += (x1 - x0) * (y1 - y0)
        return float(total)


def _as_rect(r):
    if len(r) != 4:
        raise ValueError(
            f"A footprint rectangle needs 4 numbers (xmin, xmax, ymin, ymax); got {len(r)}."
        )
    xmin, xmax, ymin, ymax = (float(v) for v in r)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(
            f"Rectangle ({xmin:g}, {xmax:g}, {ymin:g}, {ymax:g}) has non-positive "
            "width or height. The expected order is (xmin, xmax, ymin, ymax) — "
            "note it is not (xmin, ymin, xmax, ymax)."
        )
    return (xmin, xmax, ymin, ymax)


def _check_holes(holes, bbox):
    """Validate holes against the bounding box they are cut from."""
    xlo, xhi, ylo, yhi = bbox
    for h in holes:
        hx0, hx1, hy0, hy1 = h
        if hx1 <= xlo or hx0 >= xhi or hy1 <= ylo or hy0 >= yhi:
            raise ValueError(
                f"Hole ({hx0:g}, {hx1:g}, {hy0:g}, {hy1:g}) lies entirely "
                f"outside the bounding box ({xlo:g}, {xhi:g}, {ylo:g}, {yhi:g}), "
                "so it removes nothing. The expected order is "
                "(xmin, xmax, ymin, ymax) — note it is not (xmin, ymin, xmax, ymax)."
            )


def normalize_footprint(fp, *, bbox=None, mask=None, holes=None):
    """Coerce user input into a :class:`Footprint`.

    Accepts a ``Footprint``, a single rectangle, or a sequence of rectangles.
    ``mask`` may be supplied alongside a ``bbox`` for the explicit-mask case.
    ``holes`` is a sequence of rectangles to remove from the sampled area; with
    ``fp=None`` it describes a plot that is "the bounding box, minus these".
    """
    holes = tuple(_as_rect(h) for h in holes) if holes else None

    if isinstance(fp, Footprint):
        if holes is None:
            return fp
        if fp.holes is not None and fp.holes != holes:
            raise ValueError(
                f"This footprint already carries {len(fp.holes)} hole(s); pass "
                "holes= or a Footprint that has them, not both."
            )
        _check_holes(holes, fp.bbox)
        return _dc_replace(fp, holes=holes)

    if mask is not None:
        if bbox is None:
            raise ValueError("An explicit mask also needs a bbox (xmin, xmax, ymin, ymax).")
        if holes is not None:
            raise ValueError(
                "An explicit mask already encodes which patches are excluded, so "
                "holes= would be applied twice. Drop one of them."
            )
        return Footprint(bbox=_as_rect(bbox), rects=None,
                         mask=np.asarray(mask, dtype=bool).ravel())

    if fp is None:
        if bbox is None:
            raise ValueError(
                "A footprint is required: either (xmin, xmax, ymin, ymax) for a "
                "rectangular plot, or a list of rectangles for an irregular one. "
                "The sampled region cannot be inferred from stem positions — the "
                "outermost stems undershoot the true plot edges, and patch "
                "capacity depends on the true patch area."
                + ("" if holes is None else " holes= says what to remove from "
                   "the bounding box, so the bounding box itself is still needed.")
            )
        box = _as_rect(bbox)
        if holes is not None:
            _check_holes(holes, box)
        return Footprint(bbox=box, holes=holes)

    arr = list(fp)
    # A single rectangle: 4 scalars.
    if len(arr) == 4 and all(np.isscalar(v) or isinstance(v, (int, float, np.number)) for v in arr):
        box = _as_rect(arr)
        if bbox is not None:
            box = _as_rect(bbox)
        if holes is not None:
            _check_holes(holes, box)
        return Footprint(bbox=box, holes=holes)

    rects = tuple(_as_rect(r) for r in arr)
    if not rects:
        raise ValueError("The footprint rectangle list is empty.")
    # Holes are subtractive and must never grow the box, so the bbox comes from
    # the rectangles alone.
    box = (min(r[0] for r in rects), max(r[1] for r in rects),
           min(r[2] for r in rects), max(r[3] for r in rects))
    if bbox is not None:
        box = _as_rect(bbox)
    if holes is not None:
        _check_holes(holes, box)
    return Footprint(bbox=box, rects=rects, holes=holes)


def grid_shape(footprint, L, *, tiling=None):
    """Patch-grid dimensions for side length ``L``.

    ``tiling='round'`` (the default for rectangular plots) picks
    ``nx = round(W/L)`` and divides the plot exactly, so any ``L`` tiles cleanly
    and the realised patch side may differ slightly from the request.

    ``tiling='exact'`` (the default when a shape is given) requires ``L`` to
    divide the bounding box exactly, so the grid lines can align with the
    footprint edges.
    """
    W, H = footprint.width, footprint.height
    L = float(L)
    if L <= 0:
        raise ValueError(f"Patch side length must be positive; got L={L:g}.")

    if tiling is None:
        # `has_shape`, not `rects is None`: a holes-only footprint has no rects
        # but still needs grid lines landing on its hole edges, or the mask
        # depends on how W/L happened to round.
        tiling = "round" if not footprint.has_shape else "exact"

    if tiling == "round":
        nx = max(1, int(round(W / L)))
        ny = max(1, int(round(H / L)))
    elif tiling == "exact":
        nx_f, ny_f = W / L, H / L
        if abs(nx_f - round(nx_f)) > 1e-9 or abs(ny_f - round(ny_f)) > 1e-9:
            raise ValueError(
                f"L={L:g} does not divide the plot exactly (W={W:g}, H={H:g}). "
                "An irregular footprint needs grid lines that land on the shape's "
                "edges; use one of the admissible scales "
                f"({', '.join(f'{v:g}' for v in admissible_scales(footprint))})."
            )
        nx, ny = int(round(nx_f)), int(round(ny_f))
    else:
        raise ValueError(f"Unknown tiling {tiling!r}; expected 'round' or 'exact'.")

    return nx, ny, W / nx, H / ny


def patch_centers(footprint, nx, ny):
    """Centre coordinates of every grid cell, in grid order (x fastest)."""
    xmin, xmax, ymin, ymax = footprint.bbox
    dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
    ix = np.arange(nx * ny) % nx
    iy = np.arange(nx * ny) // nx
    return xmin + (ix + 0.5) * dx, ymin + (iy + 0.5) * dy


def valid_mask(footprint, nx, ny):
    """Boolean validity per grid cell, by the patch-centre rule."""
    N = nx * ny

    if footprint.mask is not None:
        m = np.asarray(footprint.mask, dtype=bool).ravel()
        if m.size != N:
            raise ValueError(
                f"The supplied mask has {m.size} entries but this grid has "
                f"{N} patches ({nx} x {ny}). A mask is tied to one spatial "
                "scale; rebuild it for this grid, or give the footprint as "
                "rectangles so it can be derived at any scale."
            )
        return m

    if not footprint.has_shape:
        return np.ones(N, dtype=bool)

    cx, cy = patch_centers(footprint, nx, ny)
    return footprint.contains(cx, cy)


def inside_footprint(footprint, x, y):
    """Point-in-footprint test (half-open), for raw stem coordinates."""
    return footprint.contains(x, y)


def _edge_gcd(footprint):
    """Largest grid spacing whose lines land on every footprint edge.

    Offsets are measured from the bounding-box origin. Returns ``None`` when the
    edges are not on a common integer lattice (e.g. fractional coordinates), in
    which case no exact ladder can be derived.
    """
    xmin, xmax, ymin, ymax = footprint.bbox
    offsets = [footprint.width, footprint.height]
    # Hole edges count too: a scale that puts a grid line through the middle of
    # a hole makes the mask depend on a float comparison at a patch centre.
    for rx0, rx1, ry0, ry1 in footprint.lattice_rects:
        offsets += [rx0 - xmin, rx1 - xmin, ry0 - ymin, ry1 - ymin]

    ints = []
    for o in offsets:
        o = abs(float(o))
        if o == 0:
            continue
        if abs(o - round(o)) > 1e-9:
            return None
        ints.append(int(round(o)))
    if not ints:
        return None
    g = 0
    for v in ints:
        g = gcd(g, v)
    return g or None


#: Fewest whole-metre sizes a rectangle's ladder may offer before it falls back
#: to exact fractional sizes (e.g. 100/6 m on a 100 x 100 plot).
MIN_WHOLE_SCALES = 3


def _thin_geometric(scales, max_scales):
    """At most ``max_scales`` of ``scales``, spread evenly on a log axis.

    The smallest and largest are always kept; each geometric target in between
    takes the nearest available size.
    """
    scales = sorted(scales)
    if len(scales) <= max_scales:
        return scales
    logs = np.log(scales)
    targets = np.linspace(logs[0], logs[-1], int(max_scales))
    return sorted({scales[int(np.argmin(np.abs(logs - t)))] for t in targets})


def admissible_scales(footprint, *, min_L=10.0, min_patches=25, max_scales=8):
    """Candidate patch side lengths ``L`` for the scale scan.

    Every size offered tiles the plot into whole, square patches: ``L`` divides
    both sides of the bounding box and, for an irregular footprint, every edge
    of its rectangles **and holes** too, so both spellings of a shape yield the
    same ladder, derived from the geometry alone. Sizes run from ``min_L`` up to
    the coarsest still giving ``min_patches`` patches, thinned to at most
    ``max_scales`` spread evenly on a log axis.

    Whole-metre sizes are preferred. A plain rectangle with too few of them
    (fewer than ``MIN_WHOLE_SCALES``) is offered the exact fractional sizes
    ``gcd(W, H) / k`` instead; one whose sides are not whole metres gets a
    geometric ladder snapped to whole column counts, whose patches may be
    slightly off square.
    """
    W, H = footprint.width, footprint.height
    L_max = float(np.sqrt(W * H / float(min_patches)))
    if L_max < float(min_L):
        raise ValueError(
            "Plot too small for the scan: even L=%g gives fewer than %d patches "
            "(plot %g x %g). Lower min_L/min_patches or pass candidate_Ls."
            % (min_L, min_patches, W, H)
        )

    def enough_patches(L):
        return round(W / L) * round(H / L) >= min_patches

    g = _edge_gcd(footprint)
    whole = [] if g is None else [
        float(L) for L in range(int(np.ceil(min_L)), g + 1)
        if g % L == 0 and enough_patches(L)]

    if footprint.has_shape or footprint.mask is not None:
        if g is None:
            raise ValueError(
                "Cannot derive a scale ladder for this footprint: its edges do "
                "not lie on a common lattice. Pass candidate_Ls explicitly."
            )
        if not whole:
            raise ValueError(
                f"No admissible scale for this footprint: L must divide {int(g)} "
                f"(the footprint edge lattice) and give at least {min_patches} "
                f"patches with L >= {min_L:g}. Pass candidate_Ls explicitly."
            )
        return _thin_geometric(whole, max_scales)

    if len(whole) >= MIN_WHOLE_SCALES:
        return _thin_geometric(whole, max_scales)
    if g is not None:
        exact = {round(g / k, 4) for k in range(1, int(g / min_L) + 1)
                 if g / k >= min_L - 1e-9 and enough_patches(g / k)}
        if exact:
            return _thin_geometric(exact, max_scales)

    raw = np.geomspace(float(min_L), L_max, int(max_scales))
    scales = {round(W / max(1, int(round(W / L))), 4) for L in raw}
    return sorted(scales)


def check_scale(footprint, L, *, min_patches=25, rtol=1e-3):
    """Check a patch side ``L`` of the user's choosing; return the side used.

    ``L`` must tile the bounding box into whole, square patches — to within
    ``rtol``, so a typed ``16.67`` stands for ``100/6`` — and, for an irregular
    footprint, land on every edge of its rectangles and holes. The grid must
    also have at least ``min_patches`` patches. Raises ``ValueError``
    saying which of these fails.

    Returns the realised side (``W / nx``), rounded as the ladder rounds it so
    that a size already on the ladder is recognised as such.
    """
    try:
        L = float(L)
    except (TypeError, ValueError):
        raise ValueError(f"Patch size must be a number of metres; got {L!r}.") from None
    if not np.isfinite(L) or L <= 0:
        raise ValueError(f"Patch size must be a positive number of metres; got {L:g}.")

    W, H = footprint.width, footprint.height
    nx, ny = int(round(W / L)), int(round(H / L))
    if nx < 1 or ny < 1:
        raise ValueError(
            f"{L:g} m is larger than the plot ({W:g} x {H:g} m).")
    side = W / nx
    if abs(side - L) > rtol * L or abs(H / ny - side) > rtol * side:
        raise ValueError(
            f"{L:g} m does not divide the plot evenly: {W:g} / {L:g} = "
            f"{W / L:.4g} and {H:g} / {L:g} = {H / L:.4g}, and both must be "
            "whole numbers for the patches to be whole squares.")

    if footprint.has_shape:
        xmin, _, ymin, _ = footprint.bbox
        for rx0, rx1, ry0, ry1 in footprint.lattice_rects:
            for o in (rx0 - xmin, rx1 - xmin, ry0 - ymin, ry1 - ymin):
                q = o / side
                if abs(q - round(q)) > 1e-6:
                    raise ValueError(
                        f"{L:g} m divides the outer range but not the outline: "
                        "grid lines would cut through the edge of an area "
                        "added or removed, so patches there would be part "
                        "surveyed.")
    else:
        side = round(side, 4)

    # Counted over the bounding box, as the ladder counts them, so a size the
    # ladder offers is never refused here.
    if nx * ny < min_patches:
        raise ValueError(
            f"{L:g} m gives only {nx * ny} patches; at least {min_patches} "
            "are needed to judge how evenly each species is spread.")
    return side
