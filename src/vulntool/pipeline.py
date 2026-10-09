from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import fit as _fit
from . import theory as _theory
from . import gof as _gof
from .footprint import normalize_footprint
from .io import (
    as_matrix,
    census_matrices,
    default_scales,
    dbh_scale_from_units,
    CROWN_ALPHA,
    CROWN_UNITS,
    STEM_UNITS,
)
from .selection import select_species, FRAC_KEEP, MAX_ZERO_FRAC
from .validate import validate_matrix


@dataclass
class VulnerabilityResult:
    """Result of :func:`compute_vulnerability`.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per species kept: ``species, abundance, p_alpha, betabar,
        delta, W_alpha, cvm``.

        * ``abundance`` — the species' total across the whole plot.
        * ``p_alpha`` — how much of an average patch it takes up, as a
          fraction. The plain "how common is it" axis.
        * ``delta`` — how evenly it is spread. Small means clumpy: plenty in a
          few patches, none in most.
        * ``betabar`` — the companion of ``delta``; together they describe the
          whole pattern of variation from patch to patch. (In the paper these
          are the shape and the dressed rate, ``delta_alpha`` and
          ``beta-bar_alpha``, of a Gamma distribution.)
        * ``W_alpha`` — the answer: how exposed the species is to being crowded
          out locally. More negative is safer.
        * ``cvm`` — how well the model matched this particular species.
    M0 : float
        How much a single patch holds when full, fitted from the data rather
        than set by the user. Everything else is measured relative to it.
    cv_xs : float
        How unequal the kept species are in typical abundance — a comparison
        *between species*, not between patches.
    Delta : float
        How far the fitted community is from internal consistency; near zero is
        what you want. Diagnostic only.
    selection : dict
        Which species were kept and dropped, and by which threshold.
    gof : float
        Overall fit score at the optimum (higher is better; it is ``-SSE``).
    cvm : float
        How far the fitted spread is from the real one, for the typical
        species. Lower is better, and below ``1/6`` counts as a good fit.
        Per-species values are the ``cvm`` column of ``table``.
    metric : str | None
        What an abundance counts: ``"stems"`` if trees were counted, or
        ``"crown"`` if they were weighted by canopy area. ``None`` when the
        input was a bare matrix, which does not say which it holds.
    units : str | None
        Units of ``abundance`` and hence of ``M0`` — ``"stems"`` for counts,
        ``"mm^(4/3)"`` for the crown metric. ``None`` when unknown; labels then
        omit the unit rather than guess it. The ``abundance`` **column name is
        deliberately not** unit-tagged, so downstream code can key on it.
    xs : (S, N) array | None
        The vacancy-adjusted abundance the Gamma was fitted to, one row per species in
        ``table`` order and one column per patch (zero where the species is
        absent). Kept so the fits can be inspected — see
        :func:`vulntool.plot_species_fits`.
    """
    table: pd.DataFrame
    M0: float
    cv_xs: float
    Delta: float
    selection: dict = field(default_factory=dict)
    gof: float = float("nan")
    cvm: float = float("nan")
    metric: object = None
    units: object = None
    xs: object = field(default=None, repr=False)

    @property
    def M0_label(self):
        """``M0`` formatted with its units, for display.

        Two significant figures: ``M0`` for the crown metric is routinely in
        the millions, and a longer mantissa left the unit truncated in the
        GUI's metric widget.
        """
        return f"{self.M0:.2g}" + (f" {self.units}" if self.units else "")

    @property
    def abundance_label(self):
        """Header for the abundance column, for display."""
        return "abundance" + (f" ({self.units})" if self.units else "")

    def to_csv(self, path, **kwargs):
        """Write the per-species table to CSV."""
        self.table.to_csv(path, index=False, **kwargs)
        return path

    def __repr__(self):
        return (f"VulnerabilityResult(species={len(self.table)}, "
                f"M0={self.M0_label}, cv_xs={self.cv_xs:.4g}, Delta={self.Delta:.4g}, "
                f"cvm={self.cvm:.4g})")


def _compute_from_matrices(fit_data, sel_data, species, *,
                           frac_keep=FRAC_KEEP, max_zero_frac=MAX_ZERO_FRAC,
                           allow_nan=False, verbose=False,
                           metric=None, units=None):
    """Core pipeline on already-loaded matrices (species x patch).

    ``fit_data`` is the matrix to fit; ``sel_data`` the matrix to select species
    on (same species order). ``species`` is the shared name list.
    """
    fit_data = validate_matrix(fit_data, species, allow_nan=allow_nan,
                               context="abundance matrix")
    if sel_data is not fit_data:
        sel_data = validate_matrix(sel_data, species, allow_nan=allow_nan,
                                   context="species-selection matrix")

    S = fit_data.shape[0]
    _, sel_idx, sel_stats = select_species(
        sel_data, frac_keep=frac_keep, max_zero_frac=max_zero_frac
    )
    if verbose:
        print(f"[select] {sel_stats}")
    if sel_idx.size == 0:
        # Distinguish "the data never arrived" from "the thresholds are strict":
        # they need opposite fixes, and the old message only ever named the latter.
        if float(np.asarray(sel_data).sum()) <= 0:
            raise ValueError(
                f"Every one of the {S} species has zero total abundance, so "
                "none can be selected. The matrix reached this point empty — "
                "check that stems were not all dropped by the plot extent or "
                "footprint (a coordinate-unit or origin mismatch does this)."
            )
        raise ValueError(
            f"No species passed the selection filter ({S} species in, "
            f"{sel_stats['S_after_frac']} kept by frac_keep={frac_keep}, then "
            f"0 left after max_zero_frac={max_zero_frac}). Every candidate is "
            "absent from more than "
            f"{100 * max_zero_frac:.0f}% of patches — loosen max_zero_frac, or "
            "use a coarser spatial scale so species occupy more patches."
        )

    fit_sel = fit_data[sel_idx]
    sel_names = [species[i] for i in sel_idx]

    if fit_data is not sel_data:
        # A species selected on sel_data (typically stem counts) but with zero
        # total on fit_data (typically crown) means every one of its stems had
        # a non-finite fit value -- e.g. DBH missing throughout, not just for
        # its dead stems. Fitting it would silently read as "absent from every
        # patch" rather than flagging the data gap that caused it.
        zero_fit = fit_sel.sum(axis=1) == 0
        if zero_fit.any():
            bad = [sel_names[i] for i in np.flatnonzero(zero_fit)]
            raise ValueError(
                f"{len(bad)} selected species have zero total in the fit "
                f"matrix despite passing selection on the other matrix: "
                f"{bad}. This means the fit value (e.g. DBH) is missing for "
                "every one of that species' stems, even though it has enough "
                "stems to be selected -- fitting it would silently treat it "
                "as absent from every patch. Fix the missing values for "
                "these species, or exclude them, before fitting."
            )

    M0, betas, deltas, gof = _fit.fit_homogeneous_M(fit_sel, verbose=verbose)
    derived = _theory.compute_derived(fit_sel, M0, betas, deltas)
    med_cvm, cvm_per = _gof.median_cvm(fit_sel, M0, betas, deltas)
    xs, _ = _theory.get_xs_from_data(fit_sel, M0)

    table = pd.DataFrame({
        "species": sel_names,
        "abundance": fit_sel.sum(axis=1),
        "p_alpha": derived["p_alpha"],
        "betabar": betas,   # paper's rate parameter beta-bar_alpha (fitted Gamma rate)
        "delta": deltas,
        "W_alpha": derived["W_alpha"],
        "cvm": cvm_per,
    })

    return VulnerabilityResult(
        table=table,
        M0=float(M0),
        cv_xs=float(derived["cv_xs"]),
        Delta=float(derived["Delta"]),
        selection=sel_stats,
        gof=float(gof),
        cvm=float(med_cvm),
        metric=metric,
        units=units,
        xs=xs,
    )


def _is_autonamed(species):
    """True when these are placeholder names, not names that came with the data.

    Unlabelled inputs (a bare array, a headerless matrix) are named ``sp0, sp1,
    ...`` on the way in, so two such inputs always agree by construction and
    there is nothing to check.
    """
    return list(species) == [f"sp{i}" for i in range(len(species))]


def _check_species_match(species, sel_species):
    """Refuse a selection matrix whose species do not line up with the fit matrix.

    The two are matched **row by row**, so a table carrying the same species in a
    different column order would select the wrong ones without failing — the kind
    of mistake that produces a plausible answer rather than an error.
    """
    if _is_autonamed(species) or _is_autonamed(sel_species):
        return
    if list(species) == list(sel_species):
        return

    same_set = set(species) == set(sel_species)
    if same_set:
        detail = ("They hold the same species in a different order. The two "
                  "tables are matched column by column, so the order has to "
                  "agree — reorder the columns of one to match the other.")
    else:
        missing = [s for s in species if s not in set(sel_species)][:5]
        extra = [s for s in sel_species if s not in set(species)][:5]
        bits = []
        if missing:
            bits.append("missing from the selection table: "
                        + ", ".join(map(repr, missing)))
        if extra:
            bits.append("only in the selection table: " + ", ".join(map(repr, extra)))
        detail = "They hold different species (" + "; ".join(bits) + ")."
    raise ValueError(
        "source and selection_source must describe the same species. " + detail
    )


def compute_vulnerability(
    source,
    *,
    selection_source=None,
    frac_keep=FRAC_KEEP,
    max_zero_frac=MAX_ZERO_FRAC,
    allow_nan=False,
    verbose=False,
):
    """Compute per-species vulnerability (W_alpha) from a census matrix.

    Parameters
    ----------
    source : str | path | DataFrame | ndarray
        The abundance matrix to fit (species x patch). CSV paths use the
        canonical header format; see :mod:`vulntool.io`.
    selection_source : same types, optional
        A separate abundance matrix used only for *selecting* species (must
        have the same species, in the same column order, as ``source``). Use
        this to select species by one metric (e.g. tree counts) while fitting
        the vulnerability model on another (e.g. crown area).

        **If omitted, species are selected from** ``source`` **itself**, which
        is the paper's policy only when ``source`` holds tree counts. A
        size-weighted table selected on itself keeps a different set of
        species, since one large tree can carry a species that few individuals
        would.
    frac_keep : float
        Cumulative-abundance fraction retained in selection (default 0.95).
    max_zero_frac : float
        Maximum empty-patch fraction allowed in selection (default 0.90).
    verbose : bool
        Print progress from the M scan and selection.

    Returns
    -------
    VulnerabilityResult
    """
    data, species, _patches = as_matrix(source)
    S, N = data.shape

    # Matrix used for species selection (defaults to the fit matrix itself).
    if selection_source is None:
        sel_data = data
    else:
        sel_data, sel_species, _ = as_matrix(selection_source)
        if sel_data.shape[0] != S:
            raise ValueError(
                "selection_source must have the same number of species as source "
                f"({sel_data.shape[0]} vs {S})."
            )
        _check_species_match(species, sel_species)

    return _compute_from_matrices(
        data, sel_data, species,
        frac_keep=frac_keep, max_zero_frac=max_zero_frac,
        allow_nan=allow_nan, verbose=verbose,
    )


# ------------------------------------------------------------------
# Census (raw stem list) path: choose the spatial scale, then compute
# ------------------------------------------------------------------
@dataclass
class ScaleScan:
    """Result of :func:`scan_scales` — a goodness-of-fit sweep over scales.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per candidate scale: ``L, n_patches, n_species, median_cvm``.
    recommended_L : float | None
        Smallest ``L`` whose ``median_cvm`` falls below ``reference`` (the
        paper's rule); ``None`` if no scale qualifies.
    reference : float
        The CvM reference value (``1/6``) the recommendation is judged against.
    results : dict
        ``{L: VulnerabilityResult}`` for scales that were fully computed (always
        includes ``recommended_L`` when defined).
    """
    table: pd.DataFrame
    recommended_L: object
    reference: float
    results: dict = field(default_factory=dict)

    def __repr__(self):
        rec = "none" if self.recommended_L is None else f"{self.recommended_L:g}"
        return (f"ScaleScan(scales={list(self.table['L'])}, "
                f"recommended_L={rec}, reference={self.reference:.4g})")

    def display_table(self):
        """:attr:`table` with headings written out, for showing to a person.

        ``table`` keeps short machine-readable names so scripts can key on them;
        this is the version to print.
        """
        return self.table.rename(columns={
            "L": "patch size",
            "n_patches": "patches",
            "n_species": "species",
            "median_cvm": "misfit",
        })


def _metric_and_units(dbh):
    """What an abundance means on the census path, given whether DBH was used."""
    return ("crown", CROWN_UNITS) if dbh is not None else ("stems", STEM_UNITS)


def _resolve_dbh_scale(dbh_scale, dbh_units):
    """Reconcile the two ways of stating DBH units; millimetres is the reference."""
    if dbh_units is not None:
        if dbh_scale is not None:
            raise ValueError(
                f"Give dbh_units={dbh_units!r} or dbh_scale={dbh_scale!r}, not "
                "both — they say the same thing and could disagree."
            )
        return dbh_scale_from_units(dbh_units)
    return 1.0 if dbh_scale is None else float(dbh_scale)


def scan_scales(
    stems,
    *,
    candidate_Ls=None,
    footprint=None,
    extent=None,
    x="x",
    y="y",
    species="species",
    dbh=None,
    alpha=CROWN_ALPHA,
    dbh_scale=None,
    dbh_units=None,
    tiling=None,
    frac_keep=FRAC_KEEP,
    max_zero_frac=MAX_ZERO_FRAC,
    reference=_gof.CVM_REFERENCE,
    min_patches=25,
    allow_nan=False,
    verbose=True,
):
    """Sweep patch side lengths and pick the default scale by goodness of fit.

    For each candidate ``L`` the stem list is binned into an ``L x L`` grid,
    species are selected on stem counts, the Gamma/M model is fitted (to the
    crown metric if ``dbh`` is given, else to stem counts), and the median
    CvM proxy is computed. The smallest ``L`` whose median CvM drops below
    ``reference`` (``1/6``) is recommended.

    Parameters
    ----------
    stems : pandas.DataFrame
        One row per stem, with coordinate, species (and optionally ``dbh``)
        columns.
    candidate_Ls : sequence of float, optional
        Scales to try. If omitted, a ladder of sizes that tile the plot into
        square patches is derived from ``extent``
        (:func:`default_scales`).
    extent : (xmin, xmax, ymin, ymax)
        Plot bounds — **required**. The sampled region cannot be inferred from
        the stem positions, and patch capacity depends on the true patch area.
    dbh : str, optional
        DBH column name. If given, the fit uses the crown metric
        (``sum of dbh**alpha`` per patch); selection still uses stem counts.
        If omitted, both use stem counts.
    reference : float
        CvM reference (default ``1/6``).
    min_patches : int
        Minimum patches for a derived candidate scale.
    verbose : bool
        Print a one-line-per-scale progress report.

    Returns
    -------
    ScaleScan
    """
    import numpy as np

    fp = normalize_footprint(footprint if footprint is not None else extent)
    dbh_scale = _resolve_dbh_scale(dbh_scale, dbh_units)
    metric, units = _metric_and_units(dbh)

    if candidate_Ls is None:
        candidate_Ls = default_scales(fp, min_patches=min_patches)
    candidate_Ls = sorted(float(L) for L in candidate_Ls)

    rows = []
    results = {}
    for L in candidate_Ls:
        fit_data, sel_data, names, binned = census_matrices(
            stems, L, x=x, y=y, species=species, dbh=dbh, alpha=alpha,
            dbh_scale=dbh_scale, footprint=fp, tiling=tiling,
        )
        res = _compute_from_matrices(
            fit_data, sel_data, names,
            frac_keep=frac_keep, max_zero_frac=max_zero_frac,
            allow_nan=allow_nan, verbose=False, metric=metric, units=units,
        )
        results[L] = res
        rows.append({
            "L": L,
            "n_patches": fit_data.shape[1],
            "n_species": len(res.table),
            "median_cvm": res.cvm,
        })
        if verbose:
            flag = "  <-- fits well" if (np.isfinite(res.cvm) and res.cvm < reference) else ""
            print(f"  {L:6g} m patches:  {fit_data.shape[1]:5d} patches, "
                  f"{len(res.table):4d} species,  misfit {res.cvm:.4f}{flag}")

    table = pd.DataFrame(rows)
    below = table[np.isfinite(table["median_cvm"]) & (table["median_cvm"] < reference)]
    recommended_L = float(below["L"].min()) if len(below) else None

    if verbose:
        if recommended_L is None:
            print(f"  No patch size got its misfit below {reference:.4g}; falling back "
                  "to the largest, which fitted best. The results are less reliable "
                  "than at a well-fitting size.")
        else:
            print(f"  Using {recommended_L:g} m patches: the smallest size that "
                  "fits well.")

    return ScaleScan(table=table, recommended_L=recommended_L,
                     reference=float(reference), results=results)


def compute_vulnerability_census(
    stems,
    *,
    L=None,
    candidate_Ls=None,
    footprint=None,
    extent=None,
    x="x",
    y="y",
    species="species",
    dbh=None,
    alpha=CROWN_ALPHA,
    dbh_scale=None,
    dbh_units=None,
    tiling=None,
    frac_keep=FRAC_KEEP,
    max_zero_frac=MAX_ZERO_FRAC,
    reference=_gof.CVM_REFERENCE,
    min_patches=25,
    allow_nan=False,
    verbose=True,
):
    """Compute vulnerability starting from a raw stem list.

    ``extent`` (plot bounds) is required. If ``L`` is given, bins at that single
    scale and computes vulnerability. Otherwise runs :func:`scan_scales` first,
    adopts the recommended scale (falling back to the coarsest scanned scale if
    none reaches the CvM reference), and returns that scale's result.

    Returns
    -------
    (VulnerabilityResult, ScaleScan | None)
        The ``ScaleScan`` is ``None`` when an explicit ``L`` was supplied.
    """
    fp = normalize_footprint(footprint if footprint is not None else extent)
    dbh_scale = _resolve_dbh_scale(dbh_scale, dbh_units)
    metric, units = _metric_and_units(dbh)
    if L is not None:
        fit_data, sel_data, names, _ = census_matrices(
            stems, float(L), x=x, y=y, species=species, dbh=dbh, alpha=alpha,
            dbh_scale=dbh_scale, footprint=fp, tiling=tiling,
        )
        res = _compute_from_matrices(
            fit_data, sel_data, names,
            frac_keep=frac_keep, max_zero_frac=max_zero_frac,
            allow_nan=allow_nan, verbose=verbose, metric=metric, units=units,
        )
        return res, None

    scan = scan_scales(
        stems, candidate_Ls=candidate_Ls, footprint=fp, x=x, y=y,
        species=species, dbh=dbh, alpha=alpha, dbh_scale=dbh_scale, tiling=tiling,
        frac_keep=frac_keep,
        max_zero_frac=max_zero_frac, reference=reference, min_patches=min_patches,
        allow_nan=allow_nan, verbose=verbose,
    )
    chosen = scan.recommended_L
    if chosen is None:
        chosen = float(scan.table["L"].max())  # coarsest = best-fitting fallback
    return scan.results[chosen], scan
