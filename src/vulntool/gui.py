"""Local Streamlit GUI for vulntool (optional; ``pip install vulntool[gui]``).

Launch with ``vulnerability gui``. Everything runs locally in the browser — no
data leaves the machine.

Covers both kinds of input: a table already divided into patches, and a raw list
of trees, for which the patch size is chosen too.

The tree-list path is a **stepper** — load the file and pick its columns,
outline the plot while watching the patches fill in, choose the scale, read the
results — with one step on screen at a time. The picture in step 2 is the point. Treating an oddly shaped plot as a rectangle
produces no error and no visible symptom in the numbers, and neither does
choosing patches so small that most come up empty — but both are obvious the
moment you see the patches drawn. So the preview keeps three things apart:
ground that was never surveyed (set aside), patches that were surveyed and hold
**nothing** (a real observation, and the one that drives which species survive
the filter), and the rest shaded by how many trees they hold.

The helpers above ``run()`` are deliberately streamlit-free so they can be
tested without the optional dependency installed; streamlit is imported inside
``run()`` and the step functions only.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np

from .footprint import (
    Footprint,
    admissible_scales,
    check_scale,
    grid_shape,
    normalize_footprint,
    patch_centers,
    valid_mask,
)
from . import __version__
from .citation import BIBTEX, CITATION_TEXT, LINK, REQUEST, TITLE, VENUE, YEAR
from .demo import DEMO_PLOT, demo_census_bytes, load_demo_patch_table
from .selection import FRAC_KEEP, MAX_ZERO_FRAC

# --------------------------------------------------------------------------
# Palette. One sequential single-hue ramp for the quantity; the two categorical
# states are drawn from outside that ramp so they can never be read as "a bit
# more" or "a bit less" of it, and both are always given a labelled swatch.
# --------------------------------------------------------------------------
#: Stem count per patch, pale yellow-green -> dark green (ColorBrewer YlGn).
SEQUENTIAL_STEPS = ["#f7fcb9", "#d9f0a3", "#addd8e", "#78c679", "#31a354", "#006837"]
#: An empty *sampled* patch — deliberately off-ramp. Purple rather than red: a
#: red marker on a green ramp is exactly the pair red-green colour blindness
#: merges, while this one stays at least ΔE 15 (OKLab x100) from every ramp
#: step and from the grey under protan, deutan and tritan simulation. It is
#: also hatched, so the state never rests on colour alone.
ZERO_COLOR = "#7b3294"
ZERO_HATCH = "///"
#: Outside the footprint: not sampled ground, so chrome rather than data.
OUTSIDE_COLOR = "#c3c2b7"
#: Sampled ground in the geometry-only preview (no data loaded yet).
SAMPLED_COLOR = "#78c679"
#: Plot outline and hole outlines.
OUTLINE_COLOR = "#0b0b0b"
#: The area being added or edited in step 2, before OK. Off the ramp and off
#: the purple; it is also drawn as a thick outline and named in the legend, so
#: it never rests on colour alone (it sits ΔE 7.7 from the darkest green under
#: deutan simulation).
DRAFT_COLOR = "#e66101"

#: The three footprint spellings the GUI offers, in radio order.
FOOTPRINT_MODES = {
    "A plain rectangle": "bbox",
    "A rectangle with pieces cut out": "holes",
    "Several blocks joined together": "rects",
}


# --------------------------------------------------------------------------
# Uploads that outlive the uploader. Streamlit drops a file_uploader's value
# on any rerun that does not render it, and the value cannot be put back
# through session_state — so a stepper that hides step 1 would lose the file
# the moment the user moved on. The bytes are kept here instead, and every
# stage reads from this wrapper, whether the file came from the uploader or
# from the bundled example.
# --------------------------------------------------------------------------
class KeptUpload:
    """A stand-in for streamlit's ``UploadedFile``, holding the file's bytes.

    It answers the four things the census path asks of an upload — ``file_id``,
    ``name``, ``size`` and ``getvalue()`` — so a kept file travels through the
    very same code as a fresh one, cache keys included: :func:`file_signature`
    of the wrapper equals that of the upload it was made from.
    """

    def __init__(self, data, name, file_id):
        self._data = bytes(data)
        self.name = name
        self.file_id = file_id

    @classmethod
    def from_upload(cls, upload):
        return cls(upload.getvalue(), upload.name,
                   getattr(upload, "file_id", "") or "")

    @property
    def size(self):
        return len(self._data)

    def getvalue(self):
        return self._data


class DemoUpload(KeptUpload):
    """The demo census. Bundled as package data, so the button works from any
    install, not just a clone of the repo."""

    name = "demo_census.csv"
    file_id = "vulntool-demo-census"

    def __init__(self, data):
        super().__init__(data, self.name, self.file_id)


#: The four number boxes of the outer box in step 2, in footprint order.
EXTENT_KEYS = ("xmin", "xmax", "ymin", "ymax")

#: Starting values for the census-mode widgets. The extent is a placeholder:
#: step 1's Next replaces it with the trees' own range. Keys match :func:`demo_settings`,
#: which overwrites them when the example is loaded.
CENSUS_DEFAULTS = {
    "fp_mode": next(iter(FOOTPRINT_MODES)),
    "xmin": 0.0,
    "xmax": 1000.0,
    "ymin": 0.0,
    "ymax": 500.0,
    "holes_text": "",
    "rects_text": "",
    "xcol": "x",
    "ycol": "y",
    "spcol": "species",
    "dbhcol": "",
    "dbh_units": "mm",
    "filter_col": "",
    "filter_vals": [],
}

#: File types the uploaders take. The separator is detected from the contents,
#: so the extension only has to get past the browser's file picker; census
#: tables are often distributed as tab- or space-separated .txt.
UPLOAD_TYPES = ["csv", "txt", "tsv"]

#: A row-filter column with more distinct values than this is almost certainly
#: not a status column (a DBH or a tag), and listing them would swamp step 1.
MAX_FILTER_VALUES = 50

#: The patch side the paper uses across all three sites; an outline that cannot
#: reach it gets a warning in step 2.
PAPER_L = 50.0
#: Fewest patches an admissible scale must give, read off ``admissible_scales``
#: so the step-2 note cannot drift from it.
MIN_PATCHES = inspect.signature(admissible_scales).parameters["min_patches"].default

#: Patch size the demo is pre-divided at for the table-input demo. Not the size
#: the census path would choose (that is the whole point of the scale scan); it
#: is a comfortable one to look at, and it fits well.
DEMO_TABLE_L = 50.0


def _rect_text(rect):
    return ", ".join(f"{float(v):g}" for v in rect)


def demo_settings():
    """The widget values that describe the demo plot, as the button applies them.

    The columns are left to :func:`guess_columns`, which finds the demo's own.

    Derived from :data:`DEMO_PLOT` rather than written out, so the outline the
    GUI fills in cannot drift from the one the demo census was made on.
    """
    holes_label = next(k for k, v in FOOTPRINT_MODES.items() if v == "holes")
    return {
        "fp_mode": holes_label,
        **dict(zip(EXTENT_KEYS,
                   (float(v) for v in DEMO_PLOT.extent))),
        "holes_text": "\n".join(_rect_text(h) for h in DEMO_PLOT.holes),
        "dbh_units": "mm",
    }


def demo_upload():
    """The demo stem list, wrapped so it can be used in place of an upload."""
    return DemoUpload(demo_census_bytes())


def parse_footprint_text(text):
    """Parse a textarea of ``xmin, xmax, ymin, ymax`` lines into rectangles.

    Blank lines and ``#`` comments are ignored. Returns ``None`` when nothing is
    specified, meaning "the whole bounding box is sampled".
    """
    rects = []
    for lineno, raw in enumerate(str(text or "").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) != 4:
            raise ValueError(
                f"Line {lineno} has {len(parts)} value(s); each rectangle needs 4: "
                "xmin xmax ymin ymax."
            )
        try:
            rects.append(tuple(float(p) for p in parts))
        except ValueError:
            raise ValueError(f"Line {lineno} contains a non-numeric value: {line!r}")
    return rects or None


def build_footprint(bbox_text, *, rects_text="", holes_text="", mode="bbox"):
    """Assemble a :class:`Footprint` from the GUI's textareas.

    ``mode`` is ``'bbox'``, ``'holes'`` (bounding box minus rectangles) or
    ``'rects'`` (union of rectangles). Holes are honoured in ``'rects'`` mode
    too — rectangles-minus-holes is a legal and occasionally handy shape.
    """
    boxes = parse_footprint_text(bbox_text)
    if boxes is None:
        raise ValueError(
            "Enter the bounding box as four numbers: xmin, xmax, ymin, ymax."
        )
    if len(boxes) > 1:
        raise ValueError(
            f"The bounding box must be a single line of four numbers; got {len(boxes)}."
        )
    bbox = boxes[0]
    holes = parse_footprint_text(holes_text)
    rects = parse_footprint_text(rects_text)

    if mode == "bbox":
        return normalize_footprint(bbox)
    if mode == "holes":
        if holes is None:
            raise ValueError(
                "This mode removes rectangles from the bounding box, so it needs "
                "at least one. Switch to 'A plain rectangle' if nothing is cut out."
            )
        return normalize_footprint(None, bbox=bbox, holes=holes)
    if mode == "rects":
        if rects is None:
            raise ValueError(
                "This mode builds the plot from rectangles, so it needs at least "
                "one. Switch to 'A plain rectangle' if the plot is a simple rectangle."
            )
        return normalize_footprint(rects, bbox=bbox, holes=holes)
    raise ValueError(f"Unknown footprint mode {mode!r}.")


def rect_label(rect):
    """One list row of step 2: ``x 0 to 100 · y 0 to 50``."""
    xmin, xmax, ymin, ymax = (float(v) for v in rect)
    return f"x {xmin:g} to {xmax:g} · y {ymin:g} to {ymax:g}"


def rects_to_text(rects):
    """The inverse of :func:`parse_footprint_text`, one rectangle per line."""
    return "\n".join(_rect_text(r) for r in rects)


def default_new_rect(bbox, side=50.0):
    """Where a newly added area starts: a ``side`` square in the box's corner.

    Placed on the plot rather than at zeros, so it shows up in the picture
    straight away; clipped so a box narrower than ``side`` still contains it.
    """
    xmin, xmax, ymin, ymax = (float(v) for v in bbox)
    return (xmin, min(xmin + side, xmax), ymin, min(ymin + side, ymax))


def rect_problem(rect, bbox, others=(), mode="holes"):
    """Why ``rect`` cannot join the step-2 list, or ``None`` if it can.

    ``others`` are the rectangles already in the list (without the one being
    edited); ``mode`` is ``'holes'`` (areas cut out) or ``'rects'`` (blocks).
    Overlaps are fine either way: blocks are joined and cut-outs are removed
    together, so only an area that would change nothing is refused.
    """
    xmin, xmax, ymin, ymax = (float(v) for v in rect)
    if xmax <= xmin or ymax <= ymin:
        return "Each 'to' must be larger than its 'from'."
    bx0, bx1, by0, by1 = (float(v) for v in bbox)
    if xmin < bx0 or xmax > bx1 or ymin < by0 or ymax > by1:
        return (f"The area must lie inside the outer range: x {bx0:g} to "
                f"{bx1:g}, y {by0:g} to {by1:g}.")
    rect = (xmin, xmax, ymin, ymax)
    others = tuple(tuple(float(v) for v in r) for r in others)
    box = (bx0, bx1, by0, by1)
    if mode == "holes":
        if others and Footprint(bbox=rect, holes=others).area() == 0:
            return "This area is already cut out by the others in the list."
        if Footprint(bbox=box, holes=others + (rect,)).area() == 0:
            return "Cutting this out would leave no surveyed ground at all."
    elif others:
        before = Footprint(bbox=box, rects=others).area()
        if Footprint(bbox=box, rects=others + (rect,)).area() == before:
            return "This area is already covered by the other blocks."
    return None


def footprint_key(footprint):
    """A hashable, canonical identity for a footprint, for use as a cache key.

    ``Footprint`` itself is unhashable when it carries a mask, and comparing two
    of them would raise on the array, so caches key on this instead.
    """
    import hashlib

    mask = footprint.mask
    return (
        tuple(float(v) for v in footprint.bbox),
        footprint.rects,
        footprint.holes,
        None if mask is None else hashlib.sha1(np.asarray(mask).tobytes()).hexdigest(),
    )


def footprint_from_key(key):
    """Rebuild a footprint from :func:`footprint_key` (masks are not restored)."""
    bbox, rects, holes, _mask = key
    return Footprint(bbox=bbox, rects=rects, holes=holes)


# --------------------------------------------------------------------------
# Which column is which. The user picks from the file's own headers; these
# pre-select the likely ones so a typical census needs no picking at all.
# --------------------------------------------------------------------------
#: Header names tried for each role, in order of preference, case-insensitively.
COLUMN_GUESSES = {
    "xcol": ("x", "gx", "px", "xcoord", "x_coord"),
    "ycol": ("y", "gy", "py", "ycoord", "y_coord"),
    "spcol": ("species", "sp", "spcode", "sp_code", "spp", "taxon", "code"),
    "dbhcol": ("dbh", "diameter"),
}


def guess_columns(columns):
    """A first guess at the x, y, species and DBH columns among ``columns``.

    Returns the four widget values keyed like :data:`COLUMN_GUESSES`. An x, y or
    species role nothing matches falls back to the first column not already
    taken, so the three always start out distinct; an unmatched DBH is ``""``
    (none) — fitting counts is a safe default, guessing a size column is not.
    """
    columns = list(columns)
    lowered = {str(c).lower(): c for c in reversed(columns)}
    guess, taken = {}, set()
    for role, names in COLUMN_GUESSES.items():
        hit = next((lowered[n] for n in names
                    if n in lowered and lowered[n] not in taken), None)
        if hit is None and role != "dbhcol":
            hit = next((c for c in columns if c not in taken), None)
        guess[role] = "" if hit is None else hit
        if hit is not None:
            taken.add(hit)
    return guess


def column_problems(stems, cols, dbhcol=""):
    """Reasons the chosen columns cannot work, as messages; empty if none."""
    import pandas as pd

    x, y, species = cols
    problems = []
    if len({x, y, species}) < 3:
        problems.append("The same column is chosen for more than one of x, y "
                        "and species.")
    if dbhcol and dbhcol in (x, y, species):
        problems.append(f"*{dbhcol}* is chosen both as DBH and as x, y or "
                        "species.")
    for role, col in (("x", x), ("y", y), ("DBH", dbhcol)):
        if col and col in stems and not pd.api.types.is_numeric_dtype(stems[col]):
            problems.append(f"The {role} column *{col}* contains values that "
                            "are not numbers.")
    return problems


def data_extent(stems, xcol, ycol, step=50.0):
    """The trees' range as ``(xmin, xmax, ymin, ymax)``, each end rounded to
    the nearest multiple of ``step``.

    Where step 2's outer box starts. Only a starting point: survey plots are
    laid out in round numbers, and the outermost trees fall a little short of
    the true edge (or, with a few stray coordinates, just past it), so the
    nearest round value is usually the edge itself. A range too narrow to
    survive rounding is widened outward to one ``step`` instead. ``None`` when
    either column holds no finite number.
    """
    lo_hi = []
    for col in (xcol, ycol):
        values = np.asarray(stems[col], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        lo = step * np.round(values.min() / step)
        hi = step * np.round(values.max() / step)
        if hi <= lo:
            lo = step * np.floor(values.min() / step)
            hi = max(step * np.ceil(values.max() / step), lo + step)
        lo_hi += [float(lo), float(hi)]
    return tuple(lo_hi)


# --------------------------------------------------------------------------
# Binned-census preview
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PatchPreview:
    """A binned census summarised per patch — what the tiling and caption need.

    ``totals`` spans the **full** ``nx * ny`` grid and is ``NaN`` outside the
    footprint, so a ``0`` in it unambiguously means "sampled ground with no
    stems on it" rather than "not part of the plot".
    """

    totals: np.ndarray
    mask: np.ndarray
    grid: tuple
    n_species: int
    n_stems: float
    n_empty: int
    max_count: float

    @property
    def n_sampled(self):
        return int(self.mask.sum())


def patch_totals(binned):
    """Per-patch total, scattered from the kept patches back onto the full grid.

    ``bin_census`` returns data for the retained patches only, plus the ``mask``
    that says where they sat; this puts them back so the tiling can draw the
    dropped cells too. Returns ``(totals, mask)`` with ``NaN`` outside.
    """
    nx, ny, _dx, _dy = binned.grid
    totals = np.full(nx * ny, np.nan, dtype=float)
    totals[binned.mask] = np.asarray(binned.data).sum(axis=0)
    return totals, binned.mask


def preview_census(stems, footprint, L, *, x="x", y="y", species="species"):
    """Bin a stem list at ``L`` and summarise it per patch — always stem COUNTS.

    Never the crown metric, even when a DBH column is available: the preview
    answers "is my outline right, and is this scale sane", and a size-weighted
    map hides an empty patch behind one big tree.

    Returns ``(PatchPreview, warnings)`` where ``warnings`` is the list of
    messages ``bin_census`` raised (typically stems outside the extent, which is
    how a coordinate-unit mismatch announces itself).
    """
    import warnings as _warnings

    from .io import bin_census

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        binned = bin_census(stems, L, footprint=footprint, x=x, y=y,
                            species=species, value=None)

    totals, mask = patch_totals(binned)
    inside = totals[mask]
    preview = PatchPreview(
        totals=totals,
        mask=mask,
        grid=binned.grid,
        n_species=len(binned.species),
        n_stems=float(np.nansum(inside)),
        n_empty=int((inside == 0).sum()),
        max_count=float(inside.max()) if inside.size else 0.0,
    )
    return preview, [str(w.message) for w in caught]


def with_extra_scale(footprint, scales, text):
    """The ladder with the user's own size added, if it tiles the outline.

    Returns ``(scales, extra, error)``: the ladder (sorted, the extra size in
    it when valid), the size actually used (``None`` if none was given or it
    does not fit), and a message saying why it does not.
    """
    text = (text or "").strip()
    if not text:
        return tuple(scales), None, None
    try:
        extra = check_scale(footprint, text.removesuffix("m").strip(),
                            min_patches=MIN_PATCHES)
    except ValueError as exc:
        return tuple(scales), None, f"{exc} It is not offered."
    return tuple(sorted({*scales, extra})), extra, None


def scale_stats(stems, footprint, scales, *, x="x", y="y", species="species"):
    """One row per candidate scale: geometry, then what the census puts in it.

    The geometry columns need no data; the rest come from one binning pass per
    scale. The share of empty patches is the column to watch: it is what the
    species filter and the fit are most sensitive to.
    """
    import pandas as pd

    rows = []
    for L in scales:
        nx, ny, dx, dy = grid_shape(footprint, L)
        mask = valid_mask(footprint, nx, ny)
        prev, _ = preview_census(stems, footprint, L, x=x, y=y, species=species)
        inside = prev.totals[mask]
        rows.append({
            "patch size": float(L),
            "grid": f"{nx} x {ny}",
            "patches surveyed": int(mask.sum()),
            "patches set aside": int(mask.size - mask.sum()),
            "patch area": float(dx * dy),
            "species": prev.n_species,
            "trees": int(prev.n_stems),
            "typical trees/patch": float(np.median(inside)) if inside.size else np.nan,
            "patches with none": prev.n_empty,
            "% with none": 100.0 * prev.n_empty / max(1, inside.size),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def counts_colormap():
    """The single-hue sequential ramp used for stem counts."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("vulntool_counts", SEQUENTIAL_STEPS)


def patch_colors(mask, values=None, *, log_scale=True):
    """Per-cell RGBA for the tiling, as an ``(N, 4)`` array.

    With ``values=None`` this is the geometry-only scheme: sampled green,
    dropped grey. Otherwise sampled patches are filled from the sequential ramp
    in proportion to ``values`` — except exact zeros, which get
    :data:`ZERO_COLOR`, because an empty sampled patch is a distinct state and
    must not be mistaken for the pale end of the ramp.

    Kept free of matplotlib artists so the colour rules can be asserted directly.
    """
    from matplotlib.colors import LogNorm, Normalize, to_rgba

    mask = np.asarray(mask, dtype=bool)
    colors = np.tile(np.asarray(to_rgba(OUTSIDE_COLOR)), (mask.size, 1))

    if values is None:
        colors[mask] = to_rgba(SAMPLED_COLOR)
        return colors

    vals = np.asarray(values, dtype=float)
    if vals.size != mask.size:
        raise ValueError(
            f"values has {vals.size} entries but this grid has {mask.size} "
            f"patches. It must span the full grid including the dropped patches "
            "— use patch_totals(binned)[0], not binned.data.sum(axis=0)."
        )

    finite = mask & np.isfinite(vals)
    colors[finite & (vals <= 0)] = to_rgba(ZERO_COLOR)

    pos = finite & (vals > 0)
    if pos.any():
        v = vals[pos]
        vmin, vmax = float(v.min()), float(v.max())
        cmap = counts_colormap()
        if vmax <= vmin:
            # One distinct value: paint it at full strength rather than let
            # Normalize map it to the near-white bottom of the ramp.
            colors[pos] = cmap(1.0)
        else:
            norm = (LogNorm(vmin, vmax) if log_scale and vmin > 0
                    else Normalize(vmin, vmax))
            colors[pos] = cmap(norm(v))
    return colors


def footprint_figure(footprint, L, ax=None, *, values=None, log_scale=True,
                     colorbar=False, draft=None):
    """Draw the patch tiling of a footprint.

    With ``values=None`` (the default) this is the pure geometry preview:
    sampled patches green, dropped patches grey.

    ``values`` is a per-patch array over the **full** ``nx * ny`` grid, ``NaN``
    outside the footprint — i.e. ``patch_totals(binned)[0]``. Sampled patches
    are then shaded by count, with empty ones flagged in :data:`ZERO_COLOR`.

    ``draft`` is a rectangle being entered in step 2, drawn on top in
    :data:`DRAFT_COLOR` without re-binning anything.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fp = normalize_footprint(footprint)
    nx, ny, dx, dy = grid_shape(fp, L)
    mask = valid_mask(fp, nx, ny)
    cx, cy = patch_centers(fp, nx, ny)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    colors = patch_colors(mask, values, log_scale=log_scale)
    empty = np.zeros(mask.size, dtype=bool)
    if values is not None:
        vals = np.asarray(values, dtype=float)
        empty = mask & np.isfinite(vals) & (vals <= 0)
    for (x, y, colour, hatched) in zip(cx, cy, colors, empty):
        ax.add_patch(Rectangle((x - dx / 2, y - dy / 2), dx, dy,
                               facecolor=colour, edgecolor="white", linewidth=0.3,
                               hatch=ZERO_HATCH if hatched else None))

    for xmin, xmax, ymin, ymax in fp.outer_rects:
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               fill=False, edgecolor=OUTLINE_COLOR, linewidth=1.8))
    for xmin, xmax, ymin, ymax in (fp.holes or ()):
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               fill=False, edgecolor=OUTLINE_COLOR, linewidth=1.2,
                               linestyle="--"))

    _draw_draft(ax, draft)
    _frame_axes(ax, fp, extra=draft)

    # No title: the counts it used to repeat are in the metric tiles above
    # the picture, and the count range is written on the colorbar's ends.
    if values is not None:
        inside = np.asarray(values, dtype=float)[mask]
        n_empty = int((inside == 0).sum())
        _add_patch_legend(ax, n_empty, draft=draft is not None)
        if colorbar:
            _add_counts_colorbar(ax, inside, log_scale=log_scale)
    return ax


def _frame_axes(ax, fp, extra=None):
    """Limits, aspect and labels shared by every drawing of a plot outline.

    ``extra`` is a rectangle to keep in view as well, so a draft that strays
    outside the outer range is still seen, and seen to stray.
    """
    xmin, xmax, ymin, ymax = fp.bbox
    if extra is not None:
        xmin, xmax = min(xmin, extra[0]), max(xmax, extra[1])
        ymin, ymax = min(ymin, extra[2]), max(ymax, extra[3])
    pad = 0.02 * max(xmax - xmin, ymax - ymin)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    # The plot outline is the frame; a box of spines around it only doubles it.
    from .plotting import tidy_axes

    tidy_axes(ax, spines=())


def _draw_draft(ax, draft):
    """The rectangle being entered: a thick outline over a light wash."""
    from matplotlib.patches import Rectangle

    if draft is None:
        return
    xmin, xmax, ymin, ymax = draft
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                           facecolor=DRAFT_COLOR, alpha=0.25, edgecolor="none",
                           zorder=5))
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
                           edgecolor=DRAFT_COLOR, linewidth=2.5, zorder=6))


def outer_range_figure(footprint, ax=None, *, draft=None):
    """The outer range alone, dashed: the frame the surveyed blocks go in."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fp = normalize_footprint(footprint)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    xmin, xmax, ymin, ymax = fp.bbox
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
                           edgecolor=OUTLINE_COLOR, linewidth=1.2,
                           linestyle="--"))
    _draw_draft(ax, draft)
    _frame_axes(ax, fp, extra=draft)
    ax.set_title(f"Outer range {xmax - xmin:g} x {ymax - ymin:g} — no blocks yet")
    if draft is not None:
        _add_patch_legend(ax, 0, draft=True, outside=False)
    return ax


def _rgb(hex_color):
    from matplotlib.colors import to_rgb
    return to_rgb(hex_color)


def figure_size_for(footprint, width=7.5, extra_height=1.4):
    """Figure size matching the plot's own aspect ratio.

    The tiling is drawn with ``aspect="equal"``, so a figure whose shape differs
    from the plot's leaves the axes box taller or wider than the drawing — and
    the colorbar, which is sized from that box, then overshoots the map. Sizing
    the figure to the footprint avoids the mismatch for any plot shape.
    """
    ratio = footprint.height / footprint.width if footprint.width else 1.0
    return (width, float(np.clip(width * ratio, 2.0, 9.0)) + extra_height)


def _add_patch_legend(ax, n_empty, *, draft=False, outside=True):
    """Label the categorical states; colour alone never carries meaning."""
    from matplotlib.patches import Patch

    handles = []
    if n_empty:
        handles.append(Patch(facecolor=ZERO_COLOR, edgecolor="white",
                             hatch=ZERO_HATCH, label="Empty patch"))
    if outside:
        handles.append(Patch(facecolor=OUTSIDE_COLOR, edgecolor="white",
                             label="Outside surveyed area"))
    if draft:
        handles.append(Patch(facecolor=(*_rgb(DRAFT_COLOR), 0.25),
                             edgecolor=DRAFT_COLOR, linewidth=2.5,
                             label="Area being entered"))
    # Below the axes, not beside them: the colorbar owns the right margin.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11),
              ncol=len(handles), frameon=False, fontsize=8)


def _add_counts_colorbar(ax, inside, *, log_scale=True):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm, Normalize

    pos = inside[np.isfinite(inside) & (inside > 0)]
    if pos.size == 0:
        return
    vmin, vmax = float(pos.min()), float(pos.max())
    if vmax <= vmin:
        return
    norm = LogNorm(vmin, vmax) if log_scale and vmin > 0 else Normalize(vmin, vmax)
    sm = ScalarMappable(norm=norm, cmap=counts_colormap())
    # The tiling is drawn with aspect="equal", so the axes box is generally
    # taller than the map inside it. A plain `colorbar(ax=ax)` is sized from
    # that box and overshoots; a divider-appended axes tracks the drawing.
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Padded clear of the map so the "min" below the bar cannot run into the
    # last x tick label.
    cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.25,
                                              axes_class=type(ax))
    cbar = ax.figure.colorbar(sm, cax=cax)
    cbar.set_label("Stems per patch")
    from .plotting import tidy_colorbar

    tidy_colorbar(cbar)
    # Plain numbers rather than 6 x 10^2, and none crowding the two ends, where
    # the actual smallest and largest count are written instead.
    from matplotlib.ticker import FixedLocator, NullLocator

    cax.yaxis.set_minor_locator(NullLocator())
    span = np.log(vmax / vmin) if isinstance(norm, LogNorm) else vmax - vmin
    pos_of = ((lambda t: np.log(t / vmin) / span) if isinstance(norm, LogNorm)
              else (lambda t: (t - vmin) / span))
    ticks = [t for t in _nice_count_ticks(vmin, vmax, log=isinstance(norm, LogNorm))
             if 0.12 < pos_of(t) < 0.88]
    cax.yaxis.set_major_locator(FixedLocator(ticks))
    cax.set_yticklabels([f"{t:,.0f}" for t in ticks])
    # The word sits centred on the bar, the number just to its right.
    style = dict(fontsize=8.5, color="#222222")
    for word, value, y, va in (("max", vmax, 1.015, "bottom"),
                               ("min", vmin, -0.015, "top")):
        label = cax.text(0.5, y, word, transform=cax.transAxes, ha="center",
                         va=va, **style)
        cax.annotate(f"{value:,.0f}", xy=(1.0, 0.0), xycoords=label,
                     xytext=(2.5, 0.0), textcoords="offset points",
                     ha="left", va="bottom", **style)


def _nice_count_ticks(vmin, vmax, *, log=True):
    """Round counts (1-2-5 series on a log bar, a linear MaxNLocator otherwise)."""
    if not log:
        from matplotlib.ticker import MaxNLocator

        return [t for t in MaxNLocator(5, integer=True).tick_values(vmin, vmax)
                if vmin < t < vmax]
    ticks = []
    for exp in range(int(np.floor(np.log10(vmin))), int(np.ceil(np.log10(vmax))) + 1):
        for m in (1, 2, 5):
            t = m * 10.0 ** exp
            if vmin < t < vmax:
                ticks.append(t)
    if len(ticks) < 3:  # a narrow range: fill in the 1-10 mantissas
        ticks = [m * 10.0 ** exp
                 for exp in range(int(np.floor(np.log10(vmin))),
                                  int(np.ceil(np.log10(vmax))) + 1)
                 for m in range(1, 10) if vmin < m * 10.0 ** exp < vmax]
    return ticks


# --------------------------------------------------------------------------
# Cacheable workers. Plain functions so the module imports without streamlit;
# ``run()`` wraps them in st.cache_data, which keys on the function's code and
# so keeps its entries across reruns.
# --------------------------------------------------------------------------
def _read_csv(_upload, file_id):
    """Parse an uploaded CSV (stem list or patch table). ``file_id`` is the cache key.

    ``_upload`` is underscore-prefixed so streamlit does not try to hash a
    possibly-huge buffer on every rerun. Note ``getvalue()`` rather than handing
    the ``UploadedFile`` to ``read_csv``: read_csv consumes the buffer to EOF,
    so a second rerun would see an empty file.

    The column separator is auto-detected (comma, tab, semicolon, ...), so a
    tab-separated file works with no extra step.
    """
    import io

    import pandas as pd

    return pd.read_csv(io.BytesIO(_upload.getvalue()), sep=None, engine="python")


def _filter_worker(_stems, file_id, row_filter):
    """The stem list with only the rows ``row_filter`` keeps (all if ``None``)."""
    from .io import keep_rows

    if row_filter is None:
        return _stems
    return keep_rows(_stems, *row_filter)


def _column_values_worker(_stems, file_id, column):
    """The distinct values of one column, as text, sorted; for the row filter."""
    from .io import column_as_text

    return sorted(column_as_text(_stems[column]).unique())


def _demo_table_worker():
    return load_demo_patch_table()


def _table_compute_worker(source, selection_source, frac_keep, max_zero_frac,
                          allow_nan):
    from .pipeline import compute_vulnerability

    return compute_vulnerability(
        source, selection_source=selection_source, frac_keep=frac_keep,
        max_zero_frac=max_zero_frac, allow_nan=allow_nan,
    )


def _preview_worker(_stems, file_id, fp_key, L, cols):
    x, y, species = cols
    return preview_census(_stems, footprint_from_key(fp_key), L,
                          x=x, y=y, species=species)


def _scale_stats_worker(_stems, file_id, fp_key, scales, cols):
    x, y, species = cols
    return scale_stats(_stems, footprint_from_key(fp_key), scales,
                       x=x, y=y, species=species)


def _compute_worker(_stems, file_id, fp_key, L, scales, cols, dbh, dbh_units,
                    frac_keep, max_zero_frac, allow_nan):
    from .pipeline import compute_vulnerability_census

    x, y, species = cols
    return compute_vulnerability_census(
        _stems, L=L, candidate_Ls=scales, footprint=footprint_from_key(fp_key),
        x=x, y=y, species=species, dbh=dbh, dbh_units=dbh_units,
        frac_keep=frac_keep, max_zero_frac=max_zero_frac,
        allow_nan=allow_nan, verbose=False,
    )


def file_signature(upload):
    """Stable identity of an uploaded file across reruns."""
    if upload is None:
        return None
    return (getattr(upload, "file_id", "") or "", upload.name, int(upload.size))


def preview_config(upload, footprint, cols):
    """Everything the binned preview depends on, as one hashable tuple.

    The DBH column and its units are deliberately absent: the preview is always
    a stem count, so editing them must not throw the user back to step 1.
    """
    return (file_signature(upload), footprint_key(footprint), tuple(cols))


# --------------------------------------------------------------------------
# The app itself. ``run()`` does the one-off setup (page, sidebar, caches) and
# then shows exactly one step of the census path — a stepper, not a scroll:
#
#     1 Load  ->  2 Outline  ->  3 Calculate  ->  4 Results
#
# driven by ``session_state["step"]``. Each step is a function, and each works
# from what the step before it stored on Next (:class:`LoadedCensus`, then
# :class:`CensusSetup`), never from that step's widgets, which are not on
# screen any more. The patch-table path is the same
# idea with two steps. Streamlit and matplotlib are imported inside these
# functions, never at module level: the helpers above must stay importable
# without the optional [gui] extra.
# --------------------------------------------------------------------------
CENSUS_STEPS = ("Load", "Outline", "Calculate", "Results")
TABLE_STEPS = ("Upload", "Results")
INPUT_MODES = ("List of individual trees", "Table of patches")

#: Widget keys that must survive reruns that do not render them: the census
#: form while another step (or the other input mode) is on screen.
PERSISTED_KEYS = (*CENSUS_DEFAULTS, "preview_L", "extra_L", "fit_L", "use_dbh")


@dataclass
class _Session:
    """What every stage needs: the cached workers and the sidebar settings."""

    read_csv: object
    filter_rows: object
    column_values: object
    cached_preview: object
    cached_scale_stats: object
    cached_compute: object
    cached_table_compute: object
    demo_table: object
    frac_keep: float
    max_zero_frac: float
    allow_nan: bool

    def stems(self, holder):
        """The stem list a :class:`LoadedCensus` or :class:`CensusSetup` works
        from: the uploaded file, less the rows its filter drops."""
        raw = self.read_csv(holder.uploaded, holder.upload_id)
        return self.filter_rows(raw, holder.upload_id, holder.row_filter)


@dataclass
class LoadedCensus:
    """What step 1 establishes: the file, and which of its columns is which."""

    uploaded: KeptUpload
    cols: tuple
    dbhcol: str
    #: ``(column, values)`` of the rows kept, or ``None`` to keep every row.
    row_filter: tuple = None

    @property
    def upload_id(self):
        return file_signature(self.uploaded)[0]

    @property
    def file_id(self):
        """Cache key of the stem list downstream: the file *and* its filter."""
        return (self.upload_id, self.row_filter)


@dataclass
class CensusSetup:
    """Everything steps 1 and 2 establish, for the steps after them."""

    uploaded: KeptUpload
    footprint: Footprint
    scales: tuple
    cols: tuple
    config: tuple
    preview_L: float
    dbhcol: str
    row_filter: tuple = None

    @property
    def upload_id(self):
        return file_signature(self.uploaded)[0]

    @property
    def file_id(self):
        """Cache key of the stem list downstream: the file *and* its filter."""
        return (self.upload_id, self.row_filter)

    @property
    def fp_key(self):
        return footprint_key(self.footprint)


def run():
    import matplotlib
    matplotlib.use("Agg")
    import streamlit as st

    st.set_page_config(page_title=f"vulntool v{__version__}", page_icon="🌳", layout="wide")
    st.markdown(
        '<style>[data-testid="stMetricValue"] {font-size: 1.4rem;}</style>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Which species to include")
        st.caption(
            "Very rare species leave too few trees to say anything reliable about "
            "how they are spread out, so they are left out of the fit."
        )
        frac_keep = st.slider(
            "Keep the commonest species, up to this share of all trees",
            0.5, 1.0, FRAC_KEEP, 0.01,
            help="At 0.95, species are kept from the commonest down until they "
                 "account for 95% of all individuals; the remaining tail is dropped.",
        )
        max_zero_frac = st.slider(
            "Drop species missing from more than this share of patches",
            0.5, 1.0, MAX_ZERO_FRAC, 0.01,
            help="A species found in only a handful of patches gives the model "
                 "almost nothing to work with, however many trees it has there.",
        )
        allow_nan = st.checkbox(
            "Treat blank cells as absences",
            key="allow_nan",
            help="Off by default. A single blank cell can quietly change which "
                 "species are kept and every number that follows, so blanks are "
                 "refused unless you confirm they really mean 'none here'.",
        )
        _footer()

    session = _Session(
        read_csv=st.cache_data(show_spinner=False, max_entries=4)(_read_csv),
        filter_rows=st.cache_data(show_spinner=False, max_entries=4)(_filter_worker),
        column_values=st.cache_data(show_spinner=False, max_entries=8)(_column_values_worker),
        cached_preview=st.cache_data(show_spinner=False, max_entries=12)(_preview_worker),
        cached_scale_stats=st.cache_data(show_spinner=False, max_entries=4)(_scale_stats_worker),
        cached_compute=st.cache_data(show_spinner=False, max_entries=4)(_compute_worker),
        cached_table_compute=st.cache_data(show_spinner=False, max_entries=4)(_table_compute_worker),
        demo_table=st.cache_data(show_spinner=False, max_entries=1)(_demo_table_worker),
        frac_keep=frac_keep, max_zero_frac=max_zero_frac, allow_nan=allow_nan,
    )

    # Streamlit discards the state of widgets a rerun does not render, so
    # moving to step 2 (or switching input modes) would empty the whole census
    # form — and, worse, leave the loaded example paired with the default
    # outline rather than its own. Re-assigning each key keeps it alive.
    for key in PERSISTED_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
    # Seeded through session_state rather than as widget defaults: the demo
    # button and the column guesser write the same keys, and streamlit warns
    # when a widget carries an inline default *and* has its value set through
    # the state API.
    for key, value in CENSUS_DEFAULTS.items():
        st.session_state.setdefault(key, value)

    # The title and blurb greet the user on the first screen only; past it,
    # every line of height goes to the step at hand. The radio is below this,
    # so its value (and the step) is read from session_state before it renders.
    if st.session_state.get("input_mode", INPUT_MODES[0]) == INPUT_MODES[1]:
        first_screen = st.session_state.get("table_step", 1) == 1
    else:
        first_screen = st.session_state.get("step", 1) == 1
    if first_screen:
        st.title(f"🌳 vulntool v{__version__}")
        st.markdown("##### Species vulnerability from spatial abundance")
        st.caption(
            f"A user-friendly implementation of the species vulnerability metric "
            f"introduced in *{TITLE}* ({VENUE}). Runs locally — no data leaves "
            "this machine."
        )

    mode = st.radio(
        "What kind of data do you have?", INPUT_MODES, key="input_mode",
        horizontal=True,
        help="A list of trees has one row per tree; the tool divides the plot "
             "into patches and selects the patch size automatically. A table of "
             "patches is the same census already divided into patches, with one "
             "row per patch.",
    )

    if mode == INPUT_MODES[1]:
        step = st.session_state.setdefault("table_step", 1)
        slot = _stepper(TABLE_STEPS, step, "table_step")
        if step > 1:
            _processing_line(table=True)
        if step == 1:
            ready = table_step_upload(session)
        else:
            ready = table_step_results(session)
        _stepper_next(slot, "table_step", step, ready)
    else:
        step = st.session_state.setdefault("step", 1)
        slot = _stepper(CENSUS_STEPS, step, "step")
        if step > 1:
            _processing_line(table=False)
        if step == 1:
            ready = step_load(session)
        elif step == 2:
            ready = step_outline(session, st.session_state["loaded"])
        elif step == 3:
            ready = step_calculate(session, st.session_state["setup"])
        else:
            ready = step_results(session, st.session_state["setup"])
        _stepper_next(slot, "step", step, ready)


def _footer():
    """The citation request, small, at the foot of the sidebar on every screen."""
    import streamlit as st

    st.divider()
    st.caption(f"If you use this tool in work you publish, please cite "
               f"[*{TITLE}*]({LINK}) (Bernardi et al., {YEAR}). The full "
               "reference is under *Citing this tool*, with the results.")


def _keep_ordered(prefix, axis, end, gap=50.0):
    """An ``on_change`` that stops a range's 'from' reaching its 'to'.

    A click on the minus of 'to' (or the plus of 'from') that would cross
    the other end is turned back to ``gap`` short of it, so the pair can
    never be reversed or empty, however fast the buttons are pressed.
    """
    import streamlit as st

    def fix():
        state = st.session_state
        lo, hi = f"{prefix}{axis}min", f"{prefix}{axis}max"
        if state.get(lo) is None or state.get(hi) is None:
            return
        if state[lo] >= state[hi]:
            if end == "min":
                state[lo] = state[hi] - gap
            else:
                state[hi] = state[lo] + gap
    return fix


def _show_figure(fig, name):
    """Show ``fig`` with buttons to save it as PNG or PDF, then close it.

    The file is drawn only when a button is pressed: streamlit runs the
    callable on click, so the many reruns of step 2 pay nothing for it. A
    closed figure can still be saved, and the closure keeps it alive.
    """
    import io

    import matplotlib.pyplot as plt
    import streamlit as st

    st.pyplot(fig)
    plt.close(fig)

    def render(fmt):
        def make():
            buf = io.BytesIO()
            fig.savefig(buf, format=fmt, dpi=300, bbox_inches="tight")
            return buf.getvalue()
        return make

    cols = st.columns([1, 1, 4])
    for col, fmt, mime in ((cols[0], "png", "image/png"),
                           (cols[1], "pdf", "application/pdf")):
        col.download_button(
            f"Save {fmt.upper()}", render(fmt), file_name=f"{name}.{fmt}",
            mime=mime, key=f"save_{name}_{fmt}", on_click="ignore",
            type="tertiary", icon=":material/download:")


def _goto(key, step):
    """An ``on_click`` that moves the stepper stored under ``key``."""
    import streamlit as st

    def move():
        st.session_state[key] = step
    return move


def _processing_line(table):
    """Name the file being worked on, under the progress line.

    Only the name: browsers do not tell a page where an upload came from.
    """
    import streamlit as st

    state = st.session_state
    if table:
        if state.get("demo_matrix"):
            what = "the example table"
        elif state.get("table_kept") is not None:
            what = f"`{state['table_kept'].name}`"
        else:
            return
        sel = state.get("table_sel_kept")
        if sel is not None:
            what += f", species chosen on the counts in `{sel.name}`"
    else:
        kept = state.get("kept_upload")
        if kept is None:
            return
        what = (f"the example census (`{kept.name}`)" if isinstance(kept, DemoUpload)
                else f"`{kept.name}`")
    st.markdown(f"Processing {what}")


def _stepper(labels, current, key):
    """The progress line, with a Back button once there is somewhere to go.

    Returns the slot for the Next button on its right (``None`` on the last
    step), filled by :func:`_stepper_next` once the step has run and knows
    whether what lies ahead still matches it.
    """
    import streamlit as st

    parts = []
    for i, label in enumerate(labels, 1):
        if i < current:
            parts.append(f":green[✓ {label}]")
        elif i == current:
            parts.append(f"**{i} · {label}**")
        else:
            parts.append(f":grey[{i} · {label}]")
    line = " → ".join(parts)
    if current == 1:
        rest, fwd = st.columns([7, 1])
    elif current == len(labels):
        back, rest = st.columns([1, 7])
        fwd = None
    else:
        back, rest, fwd = st.columns([1, 6, 1])
    if current > 1:
        back.button("← Back", key=f"back_{key}_{current}",
                    on_click=_goto(key, current - 1))
    rest.markdown(line)
    return None if fwd is None else fwd.empty()


def _stepper_next(slot, key, current, ready):
    """Fill the Next slot beside the progress line.

    Enabled only when the next step has already been reached with exactly
    what the current page holds, so its results are still to hand. It only
    moves the step: nothing the page's own button would store is touched.
    """
    if slot is None:
        return
    slot.button("Next →", key=f"fwd_{key}_{current}", disabled=not ready,
                on_click=_goto(key, current + 1))


def _loaded_key(holder):
    """What step 1 settles, as a comparable tuple (from a LoadedCensus or a
    CensusSetup)."""
    return (holder.upload_id, tuple(holder.cols), holder.dbhcol, holder.row_filter)


def _compute_key(session, setup):
    """The arguments of the census fit, as ``cached_compute`` receives them
    after the stem list."""
    import streamlit as st

    fit_L = st.session_state.get("fit_L", "auto")
    fit_L = None if fit_L == "auto" else float(fit_L)
    dbh = setup.dbhcol if setup.dbhcol and st.session_state.get("use_dbh") else None
    return (setup.file_id, setup.fp_key, fit_L, setup.scales, setup.cols, dbh,
            st.session_state["dbh_units"] if dbh else None,
            session.frac_keep, session.max_zero_frac, session.allow_nan)


def _next_button(label, key, step):
    """The primary action that moves the stepper on to ``step``."""
    import streamlit as st

    st.button(label, type="primary", key=f"next_{key}_{step}",
              on_click=_goto(key, step))


def show_results(result):
    import matplotlib.pyplot as plt
    import streamlit as st

    from .gof import CVM_REFERENCE
    from .plotting import plot_vulnerability

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Species kept", len(result.table))
    c2.metric("Patch capacity", result.M0_label,
              help="How much one patch holds when full, worked out from your "
                   "data. Not something you set — reported so you can see what "
                   "the model's crowding is measured against.")
    c3.metric("Abundance spread", f"{result.cv_xs:.3g}",
              help="How unequal the kept species are in typical abundance. "
                   "Zero would mean they are all equally common; higher means "
                   "a few dominate. This compares species with each other, not "
                   "patches with each other.")
    if np.isfinite(result.cvm):
        c4.metric("Fit quality", f"{result.cvm:.3g}",
                  delta=f"good below {CVM_REFERENCE:.3g}", delta_color="off",
                  help="How closely the fitted spread matches the real one, for "
                       "the typical species. Lower is better.")

    st.subheader("Per-species results")
    # st.dataframe headers are plain text (no MathJax), so this display
    # copy gets Unicode Greek letters; the CSV download below keeps the
    # original ASCII column names.
    shown = result.table.rename(columns={
        "abundance": result.abundance_label,
        "p_alpha": "p_α",
        "delta": "δ_α",
        "betabar": "β̄_α",
        "W_alpha": "W_α",
    })
    st.dataframe(shown, width="stretch")
    with st.expander("What these columns mean"):
        st.markdown(f"""
- **{result.abundance_label}** — the species' total across the whole plot.
- **$p_\\alpha$** — how much of an average patch it takes up, as a fraction. This is
  the plain "how common is it" axis.
- **$\\delta_\\alpha$** — how evenly it is spread. **Small means clumpy**: lots in a few
  patches, none in most. Large means spread thinly and evenly everywhere.
- **$\\bar\\beta_\\alpha$** — the companion of $\\delta_\\alpha$; together the two describe
  the whole pattern of how the species' abundance varies from patch to patch.
- **$W_\\alpha$** — the answer: how exposed the species is to being crowded out
  locally. **More negative is safer**; near or above zero is the danger zone.
  Two species equally common can land far apart here, and the clumpier one
  comes out safer — that is the point of the whole calculation.
- **cvm** — how well the model matched *this* species. Lower is better; below
  {CVM_REFERENCE:.3g} is a good fit.
""")
    st.download_button(
        "Download results CSV",
        result.table.to_csv(index=False).encode("utf-8"),
        file_name="vulnerability_results.csv", mime="text/csv",
    )

    # Asked for here rather than on the way in: the citation is only worth
    # anything to someone who has an answer in hand, and this is the moment
    # they would copy it. Nothing to dismiss.
    with st.expander("Citing this tool"):
        st.markdown(f"{REQUEST}\n\n> {CITATION_TEXT}")
        st.code(BIBTEX, language="bibtex")

    st.subheader("Vulnerability vs abundance")
    show_labels = st.checkbox(
        "Show species numbers on chart", value=True, key="show_labels",
        help="Each number is that species' row in the results table above. "
             "Off if too many species make the chart crowded.",
    )
    fig, ax = plt.subplots(figsize=(6, 4.5))
    plot_vulnerability(result, ax=ax,
                        annotate=len(result.table) if show_labels else 0)
    _show_figure(fig, "vulnerability_vs_abundance")

    show_species_fits(result)


#: Species fit panels shown at once; the rest are a page away.
FITS_PER_PAGE = 20


def show_species_fits(result):
    """Per-species fit panels, one page at a time.

    The whole section sits behind a toggle, off by default: drawing the
    panels on every rerun of the results page would add a second or two to
    each.
    """
    import io

    import streamlit as st

    from .gof import CVM_REFERENCE
    from .plotting import plot_species_fits, save_species_fits_pdf, species_fit_order

    if not st.toggle("Show per-species fits", value=False, key="show_fits"):
        return

    st.subheader("Per-species fits")
    st.caption(
        "Each panel shows the histogram of the vacancy-adjusted abundance x_α "
        "over the patches where the species is present, together with the "
        "fitted density. Patches where the species is absent are excluded from the "
        "fit; their fraction is reported in the panel, along with the fitted "
        "parameters δ_α and β̄_α and the Cramér–von Mises statistic (cvm). "
        "For a correctly specified model the expected value of cvm is "
        f"approximately {CVM_REFERENCE:.3g}, which serves as an approximate "
        "reference level. The number preceding each species name is its row "
        "in the results table. For more details on the theory and the "
        "goodness-of-fit measure, please refer to the paper's Supplementary "
        "Information."
    )

    worst_first = st.radio(
        "Order", ["Descending abundance", "Descending cvm"], horizontal=True,
        key="fits_order",
        help="Descending cvm lists first the species whose data deviate most "
             "from the fitted distribution.",
    ) == "Descending cvm"
    order = species_fit_order(result, worst_first=worst_first)
    n = len(order)
    page = 0
    if n > FITS_PER_PAGE:
        pages = range(-(-n // FITS_PER_PAGE))
        page = st.selectbox(
            "Panels", pages, key=f"fits_page_{n}",
            format_func=lambda k: (f"{k * FITS_PER_PAGE + 1}–"
                                   f"{min((k + 1) * FITS_PER_PAGE, n)} of {n}"),
        )
    rows = order[page * FITS_PER_PAGE:(page + 1) * FITS_PER_PAGE]
    _show_figure(plot_species_fits(result, rows, ncols=4), "species_fits")

    def all_pages():
        buf = io.BytesIO()
        save_species_fits_pdf(result, buf, worst_first=worst_first)
        return buf.getvalue()

    st.download_button(
        f"Save all {n} species (PDF)", all_pages, file_name="species_fits_all.pdf",
        mime="application/pdf", key="save_species_fits_all", on_click="ignore",
        type="tertiary", icon=":material/download:")


def _keep_upload(uploaded, key):
    """Remember ``uploaded`` under ``key`` and return whatever is in hand.

    A fresh upload always wins and replaces what was kept; with the uploader
    empty (not rendered on the last rerun, or never used) the kept file stands.
    """
    import streamlit as st

    if uploaded is not None:
        kept = st.session_state.get(key)
        if kept is None or file_signature(kept) != file_signature(uploaded):
            st.session_state[key] = KeptUpload.from_upload(uploaded)
    return st.session_state.get(key)


# ------------------------------------------------------------------ matrix
def table_step_upload(session):
    import streamlit as st

    st.subheader("Step 1 — upload the table")
    st.markdown(
        "**Input format:** a CSV with one row per patch — species names along "
        "the top, a patch label in the first column, and how much of each "
        "species that patch holds in the cells."
    )
    uploaded = st.file_uploader("Census matrix (CSV or text)", type=UPLOAD_TYPES,
                                key="matrix_file")

    def load_demo_table():
        st.session_state["demo_matrix"] = True
        st.session_state.pop("table_kept", None)

    st.button("Load an example table", key="demo_matrix_btn",
              on_click=load_demo_table,
              help="A simulated 20-hectare forest, bundled with the app — "
                   "no download or data of your own needed.")

    # Which species to keep is decided from the table you upload. That is the
    # right thing when the cells are tree counts, but a size-weighted table
    # selected on itself keeps a different set, since one large tree can carry
    # a species that few individuals would. Nothing in the numbers says which
    # kind of table this is, so the second table has to be offered rather than
    # inferred. The tree-list mode has no such problem: it always counts trees
    # for the selection step, whether or not diameters were given.
    with st.expander("My table is size-weighted (e.g. total canopy area)"):
        st.markdown(
            "Species are picked out by how common they are, and by default "
            "that is judged on the table above. If its cells are **counts of "
            "trees**, there is nothing to do here.\n\n"
            "If instead they are size-weighted — total canopy area, basal "
            "area, biomass — you can have the choice of species made on tree "
            "counts while the vulnerability itself is still fitted to the "
            "size-weighted figures, which is what the paper does. Upload the "
            "matching table of counts below: same species, same columns, in "
            "the same order."
        )
        sel_uploaded = st.file_uploader(
            "Tree counts, for choosing the species (optional)",
            type=UPLOAD_TYPES, key="matrix_sel_file",
        )
        sel_kept = _keep_upload(sel_uploaded, "table_sel_kept")
        if sel_kept is not None and sel_uploaded is None:
            st.caption(f"Using *{sel_kept.name}* for choosing the species.")
            st.button("Forget this table", key="forget_sel_btn",
                      on_click=lambda: st.session_state.pop("table_sel_kept", None))

    if uploaded is not None:
        st.session_state.pop("demo_matrix", None)
    kept = _keep_upload(uploaded, "table_kept")

    source = None
    if st.session_state.get("demo_matrix"):
        with st.spinner("Loading the example forest…"):
            source = session.demo_table()
        st.success(
            f"Example loaded: {len(source)} patches of "
            f"{DEMO_TABLE_L:g} x {DEMO_TABLE_L:g} m, "
            f"{source.shape[1] - 1} species, "
            f"{int(source.iloc[:, 1:].to_numpy().sum()):,} trees."
        )
        st.dataframe(source.head(), width="stretch", hide_index=True)
        st.download_button(
            "Download this example as CSV",
            source.to_csv(index=False).encode("utf-8"),
            file_name="demo_patch_table.csv", mime="text/csv",
        )
    elif kept is not None:
        try:
            source = session.read_csv(kept, kept.file_id)
        except ValueError as exc:
            st.error(str(exc))
            return
        if uploaded is None:
            st.caption(f"Using *{kept.name}* ({len(source)} rows). Upload "
                       "another file to replace it.")
            st.button("Forget this file", key="forget_table_btn",
                      on_click=lambda: st.session_state.pop("table_kept", None))

    if source is None:
        st.info("Upload a census matrix to begin, or load the example.")
        return

    sel_source = None
    sel_kept = st.session_state.get("table_sel_kept")
    if sel_kept is not None:
        try:
            sel_source = session.read_csv(sel_kept, sel_kept.file_id)
        except ValueError as exc:
            st.error(str(exc))
            return

    source_key = (
        "demo" if st.session_state.get("demo_matrix") else file_signature(kept),
        None if sel_kept is None else file_signature(sel_kept),
    )

    def go():
        st.session_state.update(table_source=source, table_sel=sel_source,
                                table_source_key=source_key, table_step=2)

    st.button("Compute vulnerability", type="primary", key="table_compute_btn",
              on_click=go)
    return st.session_state.get("table_computed_key") == _table_key(session, source_key)


def _table_key(session, source_key):
    """What a patch-table fit depends on: its tables and the sidebar settings."""
    return (source_key, session.frac_keep, session.max_zero_frac, session.allow_nan)


def table_step_results(session):
    import streamlit as st

    st.subheader("Step 2 — results")
    try:
        with st.spinner("Fitting model…"):
            result = session.cached_table_compute(
                st.session_state["table_source"], st.session_state["table_sel"],
                session.frac_keep, session.max_zero_frac, session.allow_nan,
            )
    except ValueError as exc:
        st.error(str(exc))
        return
    st.session_state["table_computed_key"] = _table_key(
        session, st.session_state.get("table_source_key"))
    show_results(result)


# ------------------------------------------------------- census, step 1
def step_load(session):
    """Upload the tree list and say which of its columns is which.

    Stores a :class:`LoadedCensus` and moves to step 2 on Next. The column
    choices are dropdowns over the file's own headers, pre-selected by
    :func:`guess_columns` whenever a different file comes in.
    """
    import streamlit as st

    from .io import suggest_row_filter

    st.subheader("Step 1 — load the tree list")
    st.markdown(
        "**Input format:** one row per tree, with its x/y position in metres, its species, "
        "and optionally its trunk diameter (DBH). Other columns are ignored, apart "
        "from an optional status column for leaving out dead stems. Comma-, tab- "
        "and space-separated files are all read, including census tables "
        "distributed as plain text (.txt)."
    )
    # The uploader's key carries a counter so the demo button can empty it.
    # A file_uploader's value cannot be assigned through session_state, and
    # rendering it under a fresh key is the only way to clear one.
    uploaded = st.file_uploader(
        "Tree list (CSV or text)", type=UPLOAD_TYPES,
        key=f"stem_file_{st.session_state.get('stem_file_epoch', 0)}",
    )

    def bump_uploader():
        st.session_state["stem_file_epoch"] = (
            st.session_state.get("stem_file_epoch", 0) + 1)

    def load_demo_census():
        """Fill in the example census *and* the outline it was surveyed on.

        The two belong together: a stem list says nothing about which ground was
        walked, and the example exists partly to show a plot that is not a plain
        rectangle. Otherwise the example goes through step 1 like any file,
        columns and all, so it shows how the tool is used.
        """
        st.session_state.update(demo_settings())
        st.session_state["kept_upload"] = demo_upload()
        st.session_state.pop("preview_L", None)
        st.session_state.pop("extra_L", None)
        # So a file uploaded after the example starts from its own range again.
        st.session_state.pop("extent_for", None)
        # Asking for the example while a file is loaded means the example: empty
        # the uploader, or the file would silently win and the button would look
        # like it had only rewritten the outline.
        bump_uploader()

    def forget_census():
        st.session_state.pop("kept_upload", None)
        st.session_state.pop("loaded", None)
        st.session_state.pop("setup", None)
        bump_uploader()

    st.button("Load an example census", key="demo_census_btn",
              on_click=load_demo_census,
              help="A simulated forest, bundled with the app — no data of your "
                   "own needed. It also fills in the plot outline it was "
                   "surveyed on, which is not a plain rectangle.")

    fresh = uploaded is not None
    # A real file wins, and replaces the example so the two cannot be mixed.
    uploaded = _keep_upload(uploaded, "kept_upload")

    if isinstance(uploaded, DemoUpload):
        rows = uploaded.getvalue().splitlines()[1:]
        n_alive = sum(row.rstrip().endswith(b",A") for row in rows)
        st.caption(
            f"Using the example census: {len(rows):,} stems on a 600 x 400 m plot "
            "whose north-east quarter was never surveyed, with a small treeless "
            "clearing inside the part that was. As in many census tables, "
            f"{n_alive:,} are alive and the rest are recorded as dead or missing; "
            "the filter below keeps the live ones. Its outline is already filled "
            "in for step 2. Upload a file above to switch to your own."
        )
        st.download_button("Download the example census as CSV",
                           uploaded.getvalue(), file_name=uploaded.name,
                           mime="text/csv")
    elif uploaded is not None and not fresh:
        # Kept from an earlier visit to this step: the uploader shows empty, so
        # say what is in hand and offer a way to let go of it.
        st.caption(f"Using *{uploaded.name}* ({uploaded.size:,} bytes). Upload "
                   "another file to replace it.")
        st.button("Forget this file", key="forget_census_btn",
                  on_click=forget_census)

    next_label = "Next — the plot outline"
    if uploaded is None:
        st.info("Upload your tree list to begin, or load the example.")
        st.button(next_label, type="primary", key="next_step_2", disabled=True)
        return
    try:
        stems = session.read_csv(uploaded, uploaded.file_id)
    except ValueError as exc:
        st.error(str(exc))
        return

    columns = list(stems.columns)
    # A different file brings different headers: start its dropdowns from a
    # fresh guess rather than from names that belonged to the last one.
    signature = file_signature(uploaded)
    guess = guess_columns(columns)
    if st.session_state.get("cols_file") != signature:
        suggested = suggest_row_filter(stems)
        picks = dict(guess,
                     filter_col=suggested[0] if suggested else "",
                     filter_vals=list(suggested[1]) if suggested else [])
        # Remembered so step 1 can say which choices are still the guess.
        st.session_state.update(picks, cols_file=signature, auto_picks=picks)
    for key in ("xcol", "ycol", "spcol"):
        if st.session_state.get(key) not in columns:
            st.session_state[key] = guess[key]
    if st.session_state.get("dbhcol") not in ("", *columns):
        st.session_state["dbhcol"] = ""
    if st.session_state.get("filter_col") not in ("", *columns):
        st.session_state["filter_col"] = ""

    st.markdown(f"**{len(stems):,} rows**, with columns "
                + ", ".join(f"`{c}`" for c in columns) + ". The first rows:")
    st.dataframe(stems.head(), width="stretch", hide_index=True)

    st.markdown("**Which column is which**")
    _auto_pick_warning()
    c1, c2, c3, c4 = st.columns(4)
    xcol = c1.selectbox("x position (m)", columns, key="xcol")
    ycol = c2.selectbox("y position (m)", columns, key="ycol")
    spcol = c3.selectbox("species", columns, key="spcol")
    dbhcol = c4.selectbox(
        "DBH (optional)", ["", *columns], key="dbhcol",
        format_func=lambda c: c or "(none)",
        help="With a DBH column, a species' presence in a patch is measured by "
             "the canopy area its trees cover rather than by how many there are, "
             "so one big tree counts for more than one sapling.",
    )

    cols = (xcol, ycol, spcol)
    problems = column_problems(stems, cols, dbhcol)
    row_filter = _row_filter_widgets(session, stems, uploaded, columns, problems)
    for msg in problems:
        st.error(msg)

    loaded = LoadedCensus(uploaded=uploaded, cols=cols, dbhcol=dbhcol,
                          row_filter=row_filter)

    def go():
        # A fresh load starts out using its DBH column, if it has one;
        # step 3 can switch that off.
        st.session_state.update(loaded=loaded, step=2, use_dbh=bool(dbhcol))
        # The outer box starts at the trees' range, once per file and choice of
        # x/y columns, so coming back here and on again keeps any edits. The
        # example brings its true outline instead: its trees stop short of it.
        extent_for = (signature, xcol, ycol)
        if (not isinstance(uploaded, DemoUpload)
                and st.session_state.get("extent_for") != extent_for):
            extent = data_extent(stems, xcol, ycol)
            if extent is not None:
                st.session_state.update(dict(zip(EXTENT_KEYS, extent)))
            st.session_state["extent_for"] = extent_for

    st.button(next_label, type="primary", key="next_step_2",
              disabled=bool(problems), on_click=go)
    stored = st.session_state.get("loaded")
    return (not problems and stored is not None
            and _loaded_key(stored) == _loaded_key(loaded))


#: How each automatically pre-selected widget is named in step 1's warning.
AUTO_PICK_LABELS = {"xcol": "x position", "ycol": "y position",
                    "spcol": "species", "dbhcol": "DBH"}


def _auto_pick_warning():
    """One warning listing the step-1 choices still at their automatic guess.

    Every guess is made from column names alone, so each is shown until the
    user changes it — a guess left unchecked is the likeliest way to fit the
    wrong column without any error.
    """
    import streamlit as st

    picks = st.session_state.get("auto_picks") or {}
    state = st.session_state
    items = [f"{label} = `{picks[key]}`" for key, label in AUTO_PICK_LABELS.items()
             if picks.get(key) and state.get(key) == picks[key]]
    if (picks.get("filter_col") and state.get("filter_col") == picks["filter_col"]
            and list(state.get("filter_vals", [])) == picks["filter_vals"]):
        values = ", ".join(f"`{v}`" for v in picks["filter_vals"])
        items.append(f"rows kept where `{picks['filter_col']}` is {values}")
    if items:
        st.warning("Selected automatically from the column names: "
                   + "; ".join(items) + ". Please check that these choices are "
                   "correct before continuing.")


def _row_filter_widgets(session, stems, uploaded, columns, problems):
    """Step 1's optional "keep only rows where <column> is <values>".

    Returns the ``(column, values)`` filter, or ``None`` to keep every row, and
    appends to ``problems`` anything that should hold back Next. Only an
    unambiguous marker such as ``status = A`` is chosen automatically
    (:func:`~vulntool.io.suggest_row_filter`); with no filter set, columns whose
    name suggests a stem's status are pointed out
    (:func:`~vulntool.io.status_like_columns`). Many census tables list dead and
    missing stems as rows, and counting them would be wrong without any error.
    """
    import streamlit as st

    from .io import status_like_columns

    st.markdown("**Which rows are live trees**")
    c1, c2 = st.columns([1, 3])
    filter_col = c1.selectbox(
        "Keep only rows where", ["", *columns], key="filter_col",
        format_func=lambda c: c or "(keep every row)",
        help="Many census files also list dead or missing stems. To keep only "
             "live trees, choose the column that records each stem's status and "
             "the values that mean alive.",
    )
    if not filter_col:
        flagged = status_like_columns(stems)
        if flagged:
            names = ", ".join(f"`{c}`" for c in flagged)
            st.warning(
                f"The column(s) {names} may record whether each stem is alive. "
                "No filter is set, so every row is counted as a live tree. "
                "Please check whether this is correct.")
        return None

    raw_id = file_signature(uploaded)[0]
    if stems[filter_col].nunique(dropna=False) > MAX_FILTER_VALUES:
        problems.append(
            f"*{filter_col}* has more than {MAX_FILTER_VALUES} different values, "
            "more than a status column usually has; the filter is limited to "
            f"columns with at most {MAX_FILTER_VALUES}.")
        return None
    options = session.column_values(stems, raw_id, filter_col)
    # Values chosen for another column (or file) do not apply to this one.
    chosen = [v for v in st.session_state.get("filter_vals", []) if v in options]
    st.session_state["filter_vals"] = chosen
    values = c2.multiselect("is one of", options, key="filter_vals")
    row_filter = (filter_col, tuple(values))
    kept = len(session.filter_rows(stems, raw_id, row_filter))
    if kept == 0:
        problems.append(f"No rows match the chosen values of *{filter_col}*.")
    else:
        c2.caption(f"Keeps {kept:,} of {len(stems):,} rows.")
    return row_filter


# ------------------------------------------------------- census, step 2
def step_outline(session, loaded):
    """The plot outline and the preview scale, with the patches drawn live.

    Stores a :class:`CensusSetup` and moves to step 3 on Next.
    """
    import matplotlib.pyplot as plt
    import streamlit as st

    st.subheader("Step 2 — outline the plot and check the patches")
    try:
        stems = session.stems(loaded)
    except ValueError as exc:
        st.error(str(exc))
        return

    # Description and shape across the full width, so the numbers that move
    # the picture sit right beside it.
    st.markdown("**Plot outline**")
    st.caption(
        "Which ground was actually surveyed. This cannot be guessed from the "
        "trees — the outermost ones always fall short of the true edge."
    )
    xcol, ycol = loaded.cols[:2]
    try:
        st.caption(
            f"Your trees lie within x {stems[xcol].min():,.6g} to "
            f"{stems[xcol].max():,.6g} and y {stems[ycol].min():,.6g} to "
            f"{stems[ycol].max():,.6g}. The outer range below starts there, "
            "rounded to the nearest 50; set it to the plot's true edge if "
            "that differs."
        )
    except (KeyError, TypeError):
        pass
    fp_mode_label = st.radio("Shape", list(FOOTPRINT_MODES), key="fp_mode",
                             horizontal=True)
    fp_mode = FOOTPRINT_MODES[fp_mode_label]

    form, picture = st.columns([2, 3], gap="large")
    with form:
        extent = {}
        for axis in ("x", "y"):
            st.markdown(f"{axis} range (m)")
            lo, hi = st.columns(2)
            for col, end, label in ((lo, "min", "from"), (hi, "max", "to")):
                extent[axis + end] = col.number_input(
                    label, step=50.0, format="%g", key=axis + end,
                    on_change=_keep_ordered("", axis, end))
        bbox_text = ", ".join(f"{extent[k]:g}"
                              for k in EXTENT_KEYS)
        rects_text = holes_text = ""
        draft = None
        if fp_mode in ("holes", "rects"):
            area_text, draft = _area_list(
                fp_mode, tuple(extent[k] for k in EXTENT_KEYS))
            if area_text is None:
                return
            if fp_mode == "holes":
                holes_text = area_text
            else:
                rects_text = area_text
        editing = draft is not None

        try:
            if fp_mode == "rects" and parse_footprint_text(rects_text) is None:
                # No blocks yet: draw the outer range alone, as a frame to
                # place them in, rather than an error and no picture.
                fp, scales = build_footprint(bbox_text), ()
            elif fp_mode == "holes" and parse_footprint_text(holes_text) is None:
                # Nothing cut out yet: the whole box is surveyed for now.
                fp = build_footprint(bbox_text)
                scales = tuple(admissible_scales(fp))
            else:
                fp = build_footprint(bbox_text, rects_text=rects_text,
                                     holes_text=holes_text, mode=fp_mode)
                scales = tuple(admissible_scales(fp))
        except ValueError as exc:
            st.error(str(exc))
            return
        if not scales:
            st.info("Add at least one surveyed block to see the patches. The "
                    "box drawn is the outer range set above.")
            with picture:
                _let_the_draft_settle(draft)
                fig, ax = plt.subplots(figsize=figure_size_for(fp, extra_height=0.6))
                outer_range_figure(fp, ax=ax, draft=draft)
                _show_figure(fig, "outline")
            return

        # A size typed below joins the ladder here, before the selectbox is
        # built from it; the box itself is drawn under the selectbox.
        scales, extra_L, extra_error = with_extra_scale(
            fp, scales, st.session_state.get("extra_L", ""))
        if st.session_state.pop("extra_L_changed", False) and extra_L is not None:
            st.session_state["preview_L"] = extra_L
        # A stale stored selection would raise before anything renders once the
        # footprint changes the admissible ladder.
        if st.session_state.get("preview_L") not in scales:
            st.session_state["preview_L"] = scales[_default_scale_index(scales)]
        preview_L = st.selectbox(
            "Patch size to preview", scales, key="preview_L",
            format_func=lambda v: f"{v:g} m",
            help="Patch sizes that divide this plot shape evenly, so no patch is "
                 "left straddling an edge. This choice changes the picture only — "
                 "the size actually used for the calculation is set in step 3.",
        )
        st.text_input(
            "Add a patch size (m)", key="extra_L", placeholder="e.g. 31.25",
            on_change=lambda: st.session_state.update(extra_L_changed=True),
            help="A size of your own, offered alongside the ones listed — here, "
                 "in step 3 and in the automatic search. It must tile the "
                 "outline into whole square patches, as the listed ones do.",
        )
        if extra_error:
            st.error(extra_error)
        st.info(
            "The patch sizes available in this step and the next are computed "
            "from the outline above: only sizes that tile it exactly into at "
            f"least {MIN_PATCHES} square patches are listed. Changing the outline — the "
            "main ranges or any areas added or removed — changes the available "
            "sizes."
        )
        if max(scales) < PAPER_L:
            st.warning(
                "With this outline, the largest available patch size is small "
                f"({max(scales):g} m). With small patches, many species are "
                "absent from most of them; this lowers the fit quality and causes "
                "those species to be dropped by the selection filter. Check that "
                "the ranges cover the whole surveyed plot."
            )

    with picture:
        try:
            with st.spinner("Binning…"):
                preview, warned = session.cached_preview(
                    stems, loaded.file_id, footprint_key(fp), preview_L,
                    loaded.cols)
                stats = session.cached_scale_stats(
                    stems, loaded.file_id, footprint_key(fp), scales,
                    loaded.cols)
        except (ValueError, KeyError) as exc:
            # The shape is still worth seeing: it is often what is wrong.
            st.error(str(exc))
            fig, ax = plt.subplots(figsize=figure_size_for(fp, extra_height=0.6))
            footprint_figure(fp, preview_L, ax=ax, draft=draft)
            _show_figure(fig, "outline")
            return
        for msg in warned:
            st.warning(msg)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Patches surveyed", preview.n_sampled)
        m2.metric("Patches with no trees", preview.n_empty)
        m3.metric("Species", preview.n_species)
        m4.metric("Trees placed", f"{preview.n_stems:,.0f}")

        _let_the_draft_settle(draft)
        fig, ax = plt.subplots(figsize=figure_size_for(fp))
        footprint_figure(fp, preview_L, ax=ax, values=preview.totals,
                         colorbar=True, draft=draft)
        _show_figure(fig, f"patches_{preview_L:g}m")
        st.caption(
            f"Surveyed area {fp.area():,.0f} of {fp.width * fp.height:,.0f} m² "
            "inside the bounding box. Shading is the number of trees per "
            "patch. It counts trees even when you have given a DBH column, so that "
            "a patch with almost nothing in it cannot hide behind one large tree."
        )
        if preview.n_empty:
            st.warning(
                f"{preview.n_empty} of {preview.n_sampled} surveyed patches hold no "
                "trees at all. This can be genuine, but a large share means the "
                "patches are small enough that many species look absent almost "
                "everywhere — and get dropped for it."
            )

    with st.expander("Patch sizes that divide this plot evenly"):
        st.dataframe(stats, width="stretch", hide_index=True)

    setup = CensusSetup(uploaded=loaded.uploaded, footprint=fp, scales=scales,
                        cols=loaded.cols,
                        config=preview_config(loaded.uploaded, fp, loaded.cols),
                        preview_L=preview_L, dbhcol=loaded.dbhcol,
                        row_filter=loaded.row_filter)

    def go():
        st.session_state.update(setup=setup, step=3)

    if editing:
        st.caption("Finish the area being entered — OK or Cancel — to move on.")
    st.button("Looks right — choose the patch size", type="primary",
              key="next_step_3", on_click=go, disabled=editing)
    stored = st.session_state.get("setup")
    return (not editing and stored is not None
            and (stored.config, stored.scales, _loaded_key(stored))
            == (setup.config, setup.scales, _loaded_key(setup)))


#: Seconds the step-2 picture waits, after the area being entered changes,
#: before it is redrawn.
DRAFT_SETTLE_S = 0.6


def _let_the_draft_settle(draft):
    """Hold the redraw while the area being entered is still changing.

    Every +/- click reruns the script, and a burst of them used to pile up
    half-drawn pictures. Waiting here debounces them on the server: a click
    that arrives during the wait cancels this run at its next element, so only
    the last value of a burst gets drawn. The number boxes render above this
    point, so they still follow every click at once. Opening the editor and
    every other rerun draw straight away.
    """
    import time

    import streamlit as st

    drawn = st.session_state.get("drawn_draft")
    if draft is not None and drawn is not None and draft != drawn:
        time.sleep(DRAFT_SETTLE_S)
    st.session_state["drawn_draft"] = draft


#: What the step-2 area list calls its items, by footprint mode.
AREA_WORDS = {
    "holes": dict(
        heading="Areas cut out", noun="area", add="＋ Add an area to cut out",
        empty="Nothing cut out yet: the whole outer range counts as surveyed.",
        help="Ground inside the box that was never surveyed — a lake, a road, a "
             "neighbouring property. These patches are set aside entirely, "
             "rather than being recorded as ground where nothing grew."),
    "rects": dict(
        heading="Surveyed blocks", noun="block", add="＋ Add a block",
        empty="No blocks yet.",
        help="The surveyed area is everything these cover together. They can "
             "form an L, a staircase or any other shape, and may overlap."),
}


def _area_list(mode, bbox):
    """The step-2 list of cut-outs or blocks, with its editor and text view.

    The text under ``<mode>_text`` stays the single stored value — the same
    format the command line reads — and the list, the editor and the *Edit as
    text* box all read and rewrite it. Returns ``(text, draft)``: the text, or
    ``None`` when it cannot be read (the error is shown), and the rectangle
    being entered, or ``None`` when no editor is open.
    """
    import streamlit as st

    words = AREA_WORDS[mode]
    key = f"{mode}_text"
    text = st.session_state.get(key, "")
    try:
        rects = parse_footprint_text(text) or []
        bad = None
    except ValueError as exc:
        rects, bad = [], str(exc)

    # One editor at a time, for this mode only. Its number boxes lose their
    # state if the step is left with it open, so it closes then too.
    edit = st.session_state.get("rect_edit")
    draft_keys = [f"draft_{k}" for k in EXTENT_KEYS]
    if edit is not None and (edit[0] != mode or bad
                             or not all(k in st.session_state for k in draft_keys)):
        st.session_state.pop("rect_edit", None)
        edit = None
    editing = edit is not None

    def open_editor(index):
        rect = rects[index] if index is not None else default_new_rect(bbox)

        def cb():
            st.session_state.update(dict(zip(draft_keys, map(float, rect))),
                                    rect_edit=(mode, index), rect_error=None)
        return cb

    def remove(index):
        def cb():
            st.session_state[key] = rects_to_text(
                r for i, r in enumerate(rects) if i != index)
        return cb

    def accept():
        index = edit[1]
        rect = tuple(float(st.session_state[k]) for k in draft_keys)
        others = [r for i, r in enumerate(rects) if i != index]
        problem = rect_problem(rect, bbox, others, mode)
        if problem:
            st.session_state["rect_error"] = problem
            return
        new = list(rects)
        if index is None:
            new.append(rect)
        else:
            new[index] = rect
        st.session_state[key] = rects_to_text(new)
        st.session_state.pop("rect_edit", None)

    def cancel():
        st.session_state.pop("rect_edit", None)

    st.markdown(f"**{words['heading']}**", help=words["help"])
    if bad:
        st.error(f"{bad} Correct it under *Edit as text* below.")
    elif not rects:
        st.caption(words["empty"])
    for i, rect in enumerate(rects):
        row, pen, bin_ = st.columns([8, 1, 1], vertical_alignment="center",
                                    gap="small")
        marker = " — editing" if editing and edit[1] == i else ""
        row.markdown(f"{i + 1}. {rect_label(rect)}{marker}")
        pen.button("✏️", key=f"edit_{mode}_{i}", type="tertiary",
                   help=f"Edit this {words['noun']}", disabled=editing,
                   on_click=open_editor(i))
        bin_.button("🗑️", key=f"remove_{mode}_{i}", type="tertiary",
                    help=f"Remove this {words['noun']}", disabled=editing,
                    on_click=remove(i))

    draft = None
    if editing:
        with st.container(border=True):
            st.markdown(f"**New {words['noun']}**" if edit[1] is None
                        else f"**Editing {words['noun']} {edit[1] + 1}**")
            for axis in ("x", "y"):
                st.markdown(f"{axis} range (m)")
                lo, hi = st.columns(2)
                for col, end, label in ((lo, "min", "from"), (hi, "max", "to")):
                    col.number_input(label, step=50.0, format="%g",
                                     key=f"draft_{axis}{end}",
                                     on_change=_keep_ordered("draft_", axis, end))
            st.caption("The area is outlined in orange on the picture.")
            if st.session_state.get("rect_error"):
                st.error(st.session_state["rect_error"])
            ok, no = st.columns(2)
            ok.button("OK", type="primary", key=f"ok_{mode}", on_click=accept,
                      width="stretch")
            no.button("Cancel", key=f"cancel_{mode}", on_click=cancel,
                      width="stretch")
        draft = tuple(float(st.session_state[k]) for k in draft_keys)
    else:
        st.button(words["add"], key=f"add_{mode}", disabled=bool(bad),
                  on_click=open_editor(None))

    with st.expander("Edit as text", expanded=bool(bad)):
        st.text_area(
            "One rectangle per line: xmin, xmax, ymin, ymax", key=key,
            disabled=editing, height=110,
            help="Handy for pasting a list. Comments after # are allowed here, "
                 "but adding, editing or removing an item above rewrites the "
                 "text without them.",
        )
    return (None if bad else st.session_state.get(key, "")), draft


# ------------------------------------------------------- census, step 3
def step_calculate(session, setup):
    """Choose the fit scale and what to measure abundance by, then fit."""
    import streamlit as st

    from .io import DBH_UNITS

    st.subheader("Step 3 — calculate")
    options = ("auto", *setup.scales)
    # The outline may have changed since a size was picked here.
    if st.session_state.get("fit_L") not in options:
        st.session_state["fit_L"] = "auto"
    st.radio(
        "Patch size", options, key="fit_L", horizontal=True,
        format_func=lambda v: ("Auto — the smallest that fits well"
                               if v == "auto" else f"{v:g}"),
        help="Auto tries every size below and keeps the smallest one the model "
             "still describes well — the most spatial detail you can have "
             "without the fit degrading.",
    )
    try:
        stems = session.stems(setup)
        stats = session.cached_scale_stats(stems, setup.file_id, setup.fp_key,
                                           setup.scales, setup.cols)
    except (ValueError, KeyError) as exc:
        st.error(str(exc))
        return
    with st.expander("What each patch size gives"):
        st.dataframe(stats, width="stretch", hide_index=True)

    if setup.dbhcol:
        use_dbh = st.toggle(
            f"Weight by tree size (DBH column *{setup.dbhcol}*)", key="use_dbh",
            help="On, a species' presence in a patch is measured by the canopy "
                 "area its trees cover, so one big tree counts for more than "
                 "one sapling. Off, every tree counts once.",
        )
        if use_dbh:
            st.selectbox(
                "DBH units", sorted(DBH_UNITS), key="dbh_units",
                help="Getting the units wrong will not change the vulnerability "
                     "numbers — it only changes the units the patch capacity is "
                     "reported in.",
            )
    else:
        st.caption("Counting trees. To weight them by size instead, choose a DBH "
                   "column in step 1.")
    _next_button("Compute vulnerability", "step", 4)
    return st.session_state.get("computed_key") == _compute_key(session, setup)


# ------------------------------------------------------- census, step 4
def step_results(session, setup):
    """Run the (cached) fit for the stored setup and show what came out."""
    import matplotlib.pyplot as plt
    import streamlit as st

    from .plotting import plot_gof_vs_L

    st.subheader("Step 4 — results")
    key = _compute_key(session, setup)
    try:
        stems = session.stems(setup)
        with st.spinner("Scanning scales and fitting…"):
            result, scan = session.cached_compute(stems, *key)
    except (ValueError, KeyError) as exc:
        st.error(str(exc))
        return
    # Lets step 3 offer Next back here while nothing has changed.
    st.session_state["computed_key"] = key

    if scan is not None:
        st.markdown("**Choosing the patch size**")
        st.caption(
            "Small patches keep spatial detail but leave too few trees in each for "
            "the model to describe well; large patches fit comfortably but average "
            "spatial details away. The last column scores the fit at each size: the "
            "median, across species, of the Cramér–von Mises statistic. Lower is "
            "better, and below "
            f"{scan.reference:.3g} counts as good."
        )
        st.dataframe(scan.display_table(), width="stretch", hide_index=True)
        if scan.recommended_L is None:
            st.warning(
                "No patch size fit well enough to clear the reference, so the "
                "best available was used."
            )
        else:
            st.success(f"Using {scan.recommended_L:g} m patches: the smallest size "
                       "that fits well.")
        if st.toggle("Show fit quality against patch size", value=False,
                     key="show_gof_plot"):
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_gof_vs_L(scan, ax=ax)
            _show_figure(fig, "fit_vs_patch_size")

    show_results(result)


def _default_scale_index(scales):
    """Prefer a 50 m patch size when this footprint admits it, else the middle."""
    if 50.0 in scales:
        return list(scales).index(50.0)
    return len(scales) // 2


# `streamlit run` targets `_gui_app.py`, which imports and calls `run()`.
