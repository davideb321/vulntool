"""Input/output: load abundance matrices from CSV.

Canonical format (recommended) — a header row of species names, one leading
column of patch labels, and abundance values in the cells::

    patch,SpeciesA,SpeciesB,SpeciesC
    P1,12.3,0,4.0
    P2,0,7.5,1.0

The separator is auto-detected, so a tab- or semicolon-separated file works
just as well as a comma-separated one.

Also supported: a headerless numeric-matrix layout, where the first two
columns are patch x/y coordinates and the remaining columns are per-species
abundances (unlabelled).

A minimal ``bin_census`` helper turns a raw stem list (x, y, dbh, species) into
an abundance matrix, as a convenience for users starting from field data.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .footprint import (
    Footprint,
    admissible_scales,
    grid_shape,
    normalize_footprint,
    valid_mask,
)
from .validate import check_species_names


def load_matrix(source, patch_col=0):
    """Load an abundance matrix from the canonical header format.

    Parameters
    ----------
    source : str | path | pandas.DataFrame
        CSV path or an already-loaded DataFrame. First row = species names,
        column ``patch_col`` = patch labels, remaining columns = abundances.
        The column separator is auto-detected (comma, tab, semicolon, ...).
    patch_col : int | None
        Positional index of the patch-label column (default 0). Pass ``None``
        if the file has no label column and every column is a species.

    Returns
    -------
    data : (S, N) float array   (species x patch)
    species : list[str]         species names (from the header)
    patches : list              patch labels
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source, sep=None, engine="python")

    check_species_names(df.columns)

    if patch_col is None:
        patches = list(range(len(df)))
        abund = df
    else:
        patch_name = df.columns[patch_col]
        # A numeric label column is usually a species that is about to be eaten
        # silently: the caller forgot the patch-label column entirely.
        # pd.api.types handles extension dtypes (StringDtype etc.) that
        # np.issubdtype cannot interpret.
        if pd.api.types.is_numeric_dtype(df[patch_name]):
            warnings.warn(
                f"Column {patch_name!r} is being used as patch labels but is "
                "numeric. If the file has no label column, pass patch_col=None "
                "so it is not mistaken for one (otherwise the first species is "
                "silently dropped).",
                stacklevel=2,
            )
        patches = df[patch_name].tolist()
        abund = df.drop(columns=[patch_name])

    non_numeric = [c for c in abund.columns if not pd.api.types.is_numeric_dtype(abund[c])]
    if non_numeric:
        raise ValueError(
            "Non-numeric abundance columns found: %s. Expected a header row of "
            "species names, one patch-label column, and numeric abundances."
            % non_numeric
        )

    species = list(abund.columns)
    data = abund.to_numpy(dtype=float).T  # (S, N)
    return data, species, patches


def load_bare_matrix(path, coord_cols=2, round_ints=True):
    """Load a headerless numeric-matrix abundance format.

    First ``coord_cols`` columns are patch coordinates; the rest are species
    abundances (unlabelled — species are named ``sp0, sp1, ...``).

    Parameters
    ----------
    round_ints : bool
        If True (default), round abundances to the nearest integer. Fractional
        abundances (e.g. summed canopy area) round to a negligible relative
        error for realistic values; set False to keep them exact.

    Returns ``(data (S, N), species, patches)``.
    """
    raw = np.loadtxt(path, delimiter=",")
    coords = raw[:, :coord_cols]
    data = raw[:, coord_cols:].T  # (S, N)
    if round_ints:
        data = np.rint(data)
    species = [f"sp{i}" for i in range(data.shape[0])]
    patches = [f"p{j}" for j in range(data.shape[1])]
    # patch labels from coords when 2D, e.g. "25_75"
    if coord_cols >= 2:
        patches = ["_".join(str(int(v)) if float(v).is_integer() else str(v) for v in row)
                   for row in coords]
    return data, species, patches


def as_matrix(source):
    """Coerce ``source`` to ``(data, species, patches)``.

    Accepts a CSV path (canonical header format), a DataFrame, or a raw
    ``(S, N)`` numpy array (species auto-named).
    """
    if isinstance(source, np.ndarray):
        data = np.asarray(source, dtype=float)
        if data.ndim != 2:
            raise ValueError("Array input must be 2-D (species x patch).")
        species = [f"sp{i}" for i in range(data.shape[0])]
        patches = [f"p{j}" for j in range(data.shape[1])]
        return data, species, patches
    return load_matrix(source)


@dataclass
class BinnedCensus:
    """A stem list binned onto a patch grid.

    Attributes
    ----------
    data : (S, N_valid) float array
        Abundance per species x patch, restricted to patches inside the footprint.
    species : list[str]
        Species names, sorted; those observed inside the footprint.
    patches : list[str]
        Patch labels ``"ix_iy"`` for the retained patches.
    mask : (nx*ny,) bool array
        Which cells of the full grid are inside the footprint.
    footprint : Footprint
    grid : (nx, ny, dx, dy)
        Grid dimensions and realised patch size.
    """

    data: np.ndarray
    species: list
    patches: list
    mask: np.ndarray
    footprint: "Footprint"
    grid: tuple

    @property
    def n_patches(self):
        return int(self.data.shape[1])

    @property
    def L(self):
        """Realised patch side length (``dx``; equals ``dy`` for a square grid)."""
        return float(self.grid[2])

    def __iter__(self):
        """Unpack as ``data, species, patches`` for backwards compatibility."""
        return iter((self.data, self.species, self.patches))


def bin_census(stems, L, *, footprint=None, extent=None, x="x", y="y",
               species="species", value=None, tiling=None):
    """Bin a raw stem list into an (S, N) abundance matrix over a patch grid.

    The plot is divided into a whole number of patches by *exact division*, so
    there is never a ragged remainder strip: for a rectangular plot
    ``nx = round(W/L)`` and each patch is ``W/nx`` by ``H/ny`` (the realised side
    may differ slightly from the requested ``L``). For an irregular footprint the
    grid must additionally align with the shape's edges, so ``L`` must divide the
    bounding box exactly — see :func:`vulntool.footprint.admissible_scales`.

    Every patch inside the footprint is returned, including those with no stems:
    those are genuine zeros, and a species' zero-fraction is measured against
    them. Patches *outside* the footprint are dropped entirely — they are not
    sampled ground, and counting them as empty would corrupt every fitted
    parameter.

    A footprint is **required**: the sampled region cannot be inferred reliably
    from the stem positions (the outermost stems undershoot the true plot edges),
    and patch capacity depends on the true patch area.

    Parameters
    ----------
    stems : pandas.DataFrame
        One row per stem, with coordinate and species columns.
    L : float
        Requested patch side length, in coordinate units.
    footprint : Footprint | (xmin, xmax, ymin, ymax) | list of rectangles
        The sampled area. ``extent`` is accepted as an alias for the rectangular
        case.
    x, y, species : str
        Column names for coordinates and species.
    value : str, optional
        Column to sum per patch (e.g. a precomputed crown proxy). If None, the
        abundance is a stem count. Rows with a non-finite ``value`` are skipped
        (e.g. stems with missing DBH).
    tiling : {'round', 'exact'}, optional
        Grid-fitting rule; defaults to 'round' for rectangles and 'exact' when a
        shape is given.

    Returns
    -------
    BinnedCensus
    """
    if footprint is None and extent is None:
        raise ValueError(
            "bin_census requires a footprint: (xmin, xmax, ymin, ymax) for a "
            "rectangular plot, or a list of rectangles for an irregular one."
        )
    fp = normalize_footprint(footprint if footprint is not None else extent)
    xmin, xmax, ymin, ymax = fp.bbox
    nx, ny, dx, dy = grid_shape(fp, L, tiling=tiling)

    missing = [(role, col) for role, col in
               (("x", x), ("y", y), ("species", species)) + ((("value", value),) if value else ())
               if col not in stems.columns]
    if missing:
        raise ValueError(
            "Column(s) not found in the stem list: %s. Available columns: %s. "
            "Pass the right names via the x/y/species arguments (CLI: --x, --y, "
            "--species)." % (
                ", ".join(f"{col!r} (for {role})" for role, col in missing),
                ", ".join(map(repr, stems.columns)),
            )
        )

    xv = np.asarray(stems[x], dtype=float)
    yv = np.asarray(stems[y], dtype=float)
    sp = np.asarray(stems[species])
    w = (np.asarray(stems[value], dtype=float) if value is not None
         else np.ones(len(stems), dtype=float))

    # Half-open bounds [min, max): a stem exactly on the far edge belongs to no
    # cell rather than being folded into the last one.
    in_extent = (xv >= xmin) & (xv < xmax) & (yv >= ymin) & (yv < ymax)
    if not in_extent.any() and len(stems):
        # By far the most common real-world cause is a coordinate mismatch
        # (metres vs centimetres, a local grid vs projected coordinates, or a
        # shifted origin). Report both ranges so the mismatch is self-evident.
        raise ValueError(
            "Every stem falls outside the plot extent, so the census is empty. "
            f"Stems span x=[{np.nanmin(xv):g}, {np.nanmax(xv):g}], "
            f"y=[{np.nanmin(yv):g}, {np.nanmax(yv):g}], but the extent is "
            f"x=[{xmin:g}, {xmax:g}], y=[{ymin:g}, {ymax:g}]. Check the "
            "coordinate units and origin of both."
        )
    n_outside = int((~in_extent).sum())
    if n_outside:
        warnings.warn(
            f"bin_census: dropped {n_outside} stem(s) outside the extent.",
            stacklevel=2,
        )

    # Species observed within the footprint, regardless of whether their value
    # is finite: a species whose stems all have e.g. missing DBH must still get
    # a (zero) row, so a crown matrix and a stem-count matrix binned from the
    # same stems always agree on the species set (see census_matrices).
    species_names = sorted(pd.unique(sp[in_extent]).tolist())
    sp_pos = {s: i for i, s in enumerate(species_names)}

    keep = in_extent & np.isfinite(w)
    n_novalue = int((in_extent & ~np.isfinite(w)).sum())
    if n_novalue:
        warnings.warn(
            f"bin_census: {n_novalue} stem(s) inside the extent have a "
            "non-finite value (e.g. missing DBH) and are excluded from this "
            "matrix's sums, but still count toward species presence.",
            stacklevel=2,
        )
    xv, yv, sp, w = xv[keep], yv[keep], sp[keep], w[keep]

    ix = ((xv - xmin) / dx).astype(int)
    iy = ((yv - ymin) / dy).astype(int)
    cell = iy * nx + ix  # 0 .. nx*ny-1, x fastest

    si = np.fromiter((sp_pos[s] for s in sp), dtype=int, count=len(sp))

    S, N = len(species_names), nx * ny
    data = np.zeros((S, N), dtype=float)
    if si.size:
        np.add.at(data, (si, cell), w)

    # Restrict to sampled ground. Patches outside the footprint are not empty —
    # they do not exist, and must not dilute any species' zero-fraction.
    mask = valid_mask(fp, nx, ny)
    labels = [f"{c % nx}_{c // nx}" for c in range(N)]
    data = data[:, mask]
    patch_labels = [lab for lab, ok in zip(labels, mask) if ok]

    if data.shape[1] == 0:
        raise ValueError(
            f"No patches lie inside the footprint at L={L:g} ({nx} x {ny} grid). "
            "Check that the footprint rectangles are in (xmin, xmax, ymin, ymax) "
            "order and overlap the bounding box."
        )

    return BinnedCensus(data=data, species=species_names, patches=patch_labels,
                        mask=mask, footprint=fp, grid=(nx, ny, dx, dy))


# ------------------------------------------------------------------
# Raw-census helpers for the scale-selection (census) path
# ------------------------------------------------------------------
#: Allometric exponent for crown area from trunk diameter: A_crown ~ dbh^(4/3).
CROWN_ALPHA = 4.0 / 3.0

#: DBH units the tool knows, mapped to the factor converting them to
#: millimetres — the reference unit ``crown_from_dbh`` normalises to.
DBH_UNITS = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}

#: Units of an abundance, and hence of the fitted carrying capacity M0, on each
#: path. The crown metric is a sum of ``dbh**(4/3)`` over stems in a patch, with
#: DBH normalised to millimetres, so its units are fixed regardless of what the
#: input column was recorded in.
STEM_UNITS = "stems"
CROWN_UNITS = "mm^(4/3)"


def dbh_scale_from_units(units):
    """Factor converting a DBH column in ``units`` to millimetres."""
    try:
        return DBH_UNITS[str(units).strip().lower()]
    except KeyError:
        raise ValueError(
            f"Unknown DBH units {units!r}. Known units: "
            f"{', '.join(sorted(DBH_UNITS))}. For anything else pass an explicit "
            "numeric dbh_scale (the factor converting your DBH to millimetres)."
        ) from None


def crown_from_dbh(dbh, alpha=CROWN_ALPHA, dbh_scale=1.0):
    """Size-weighted crown proxy ``(dbh * dbh_scale) ** alpha``.

    The paper fits the crown metric = per-patch sum of ``dbh ** alpha`` with the
    allometric exponent ``alpha = 4/3``. The forest-specific prefactor cancels
    in the rescaled variable, so it is omitted here.

    ``dbh_scale`` converts the input to millimetres, the reference unit (see
    :data:`DBH_UNITS` / :func:`dbh_scale_from_units`). A global unit factor
    cancels in the rescaled variable ``xs``, so it does not change ``W_alpha`` —
    but it does **not** cancel in the fitted carrying capacity ``M0``, which is
    why the units are reported alongside it rather than left implicit.
    """
    d = np.asarray(dbh, dtype=float) * float(dbh_scale)
    # A negative DBH raised to a fractional power is NaN, which would otherwise
    # be swept into the generic "non-finite value" drop with no hint of the cause.
    neg = np.isfinite(d) & (d < 0)
    if neg.any():
        raise ValueError(
            f"{int(neg.sum())} negative DBH value(s) (minimum {np.min(d[neg]):g}). "
            "DBH cannot be negative — check for sentinel values marking missing "
            "measurements, and drop or blank those rows."
        )
    return d ** float(alpha)


def column_as_text(column):
    """A column's values as text, with a blank cell as ``"nan"``.

    ``astype(str)`` alone does that on pandas 2 but keeps blanks as missing on
    pandas 3, where they would then match nothing and break sorting.
    """
    return column.astype(str).fillna("nan")


def keep_rows(stems, column, values):
    """The rows of ``stems`` whose ``column`` is one of ``values``.

    Values are compared as text, so ``"A"`` and ``1`` both work whatever type
    pandas gave the column; a blank cell reads as ``"nan"``. Used to drop dead
    or missing stems before binning: a census row is not necessarily a live tree.
    """
    keep = {str(v) for v in values}
    return stems[column_as_text(stems[column]).isin(keep)]


#: Unambiguous live-stem markers, as (column, value): a ``status`` column of
#: A(live)/D(ead)/M(issing), as in ForestGEO census tables, or ``DFstatus``,
#: which spells the same out. Matched case-insensitively on the column name,
#: first hit wins. Only these are ever selected automatically.
ALIVE_MARKERS = (("status", "A"), ("dfstatus", "alive"))

#: Column names (lowercase) that may record whether a stem is alive, in a
#: convention too varied to guess — condition codes, say. They are pointed out
#: when no filter is set, never selected.
STATUS_LIKE_NAMES = ("status", "dfstatus", "codes", "condition", "vitality",
                     "alive", "dead", "mortality")


def suggest_row_filter(stems):
    """A ``(column, (values,))`` filter that keeps only live stems, or ``None``.

    Only for an unambiguous marker (:data:`ALIVE_MARKERS`), and only when it
    would actually drop something: a status column holding nothing but live
    stems needs no filter. Other conventions (e.g. condition codes in a
    ``codes`` column) differ from site to site and are not guessed; see
    :func:`status_like_columns`.
    """
    lowered = {str(c).lower(): c for c in reversed(list(stems.columns))}
    for name, alive in ALIVE_MARKERS:
        col = lowered.get(name)
        if col is None:
            continue
        values = set(column_as_text(stems[col]))
        if alive in values and values != {alive}:
            return col, (alive,)
    return None


def status_like_columns(stems):
    """Columns whose name suggests they record whether a stem is alive.

    Matched case-insensitively against :data:`STATUS_LIKE_NAMES`. A marker
    column (:data:`ALIVE_MARKERS`) in which every row is already alive is left
    out: there is nothing in it to filter.
    """
    markers = dict(ALIVE_MARKERS)
    found = []
    for col in stems.columns:
        name = str(col).lower()
        if name not in STATUS_LIKE_NAMES:
            continue
        if name in markers and set(column_as_text(stems[col])) == {markers[name]}:
            continue
        found.append(col)
    return found


def census_matrices(stems, L, *, footprint=None, extent=None, x="x", y="y",
                    species="species", dbh=None, alpha=CROWN_ALPHA, dbh_scale=1.0,
                    tiling=None):
    """Bin a stem list into the pair of matrices the vulnerability pipeline needs.

    Always returns a **stem-count** matrix, used for species selection. If a
    ``dbh`` column is given, also returns a **crown** matrix (``sum of
    dbh**alpha`` per patch) to fit; otherwise the fit matrix is the stem-count
    matrix itself.

    Both matrices share the same species order and patch grid. A ``footprint``
    (or rectangular ``extent``) is required; see :func:`bin_census`.

    Returns ``(fit_data, sel_data, species_names, binned)`` where ``binned`` is
    the :class:`BinnedCensus` for the stem-count pass (carrying the mask and grid).
    """
    counts = bin_census(stems, L, x=x, y=y, species=species, value=None,
                        footprint=footprint, extent=extent, tiling=tiling)
    if dbh is None:
        return counts.data, counts.data, counts.species, counts

    if dbh not in stems.columns:
        raise ValueError(
            f"DBH column {dbh!r} not found in the stem list. Available columns: "
            f"{', '.join(map(repr, stems.columns))}."
        )
    stems = stems.assign(_crown=crown_from_dbh(stems[dbh], alpha=alpha, dbh_scale=dbh_scale))
    crown = bin_census(stems, L, x=x, y=y, species=species, value="_crown",
                       footprint=footprint, extent=extent, tiling=tiling)
    # Same grid and pivot => identical species order and patch labels by construction.
    assert crown.species == counts.species and crown.patches == counts.patches
    return crown.data, counts.data, counts.species, counts


def default_scales(footprint, *, min_L=10.0, min_patches=25, max_scales=8):
    """A ladder of candidate patch side lengths ``L`` for the scale scan.

    Every size offered tiles the plot into whole, square patches: it divides
    both sides of the plot (whole-metre sizes preferred) and, for an irregular
    footprint, every edge of the shape too. The ladder runs from ``min_L`` up to
    the coarsest scale still yielding ``min_patches`` patches.

    Accepts a :class:`~vulntool.footprint.Footprint`, a rectangle, or a list of
    rectangles. See :func:`vulntool.footprint.admissible_scales`.
    """
    fp = normalize_footprint(footprint)
    return admissible_scales(fp, min_L=min_L, min_patches=min_patches,
                             max_scales=max_scales)
