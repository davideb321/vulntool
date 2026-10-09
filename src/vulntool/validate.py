"""Input validation for abundance matrices.

This module exists to prevent a plausible-looking wrong answer.

It refuses to run on corrupt input, and says exactly what is wrong and how to
fix it. Ambiguities (is this matrix transposed?) are warnings, not errors.

The one deliberate escape hatch is ``allow_nan``: a user who knows their census
has missing cells and wants them treated as absences can say so explicitly.
"""
from __future__ import annotations

import warnings

import numpy as np

#: Below this many patches, a spatial abundance distribution is not estimable.
MIN_PATCHES = 4
#: Below this many patches, the fit runs but is poorly determined.
WARN_PATCHES = 10


def _name_offenders(species, idx, limit=5):
    """Render a short, readable list of offending species for an error message."""
    if species is None:
        labels = [f"row {int(i)}" for i in idx[:limit]]
    else:
        labels = [str(species[int(i)]) for i in idx[:limit]]
    extra = "" if len(idx) <= limit else f", ... (+{len(idx) - limit} more)"
    return ", ".join(labels) + extra


def check_species_names(names):
    """Reject duplicate species names before they reach pandas.

    Duplicate columns make ``df[col]`` return a DataFrame rather than a Series,
    which surfaces much later as an unhelpable
    ``AttributeError: 'DataFrame' object has no attribute 'dtype'``.
    """
    names = list(names)
    seen, dupes = set(), []
    for n in names:
        if n in seen and n not in dupes:
            dupes.append(n)
        seen.add(n)
    if dupes:
        raise ValueError(
            "Duplicate species name(s) in the input: %s. Every species column "
            "must have a distinct name; merge or rename the duplicates."
            % ", ".join(str(d) for d in dupes[:5])
        )
    return names


def validate_matrix(data, species=None, *, allow_nan=False, context="abundance matrix",
                    min_patches=MIN_PATCHES, warn_patches=WARN_PATCHES):
    """Validate an (S, N) species-by-patch abundance matrix.

    Parameters
    ----------
    data : (S, N) array
        Abundance per species (rows) x patch (columns).
    species : sequence of str, optional
        Species names, used to make error messages specific.
    allow_nan : bool
        If True, non-finite cells are replaced by zero (treated as absences)
        with a warning instead of raising.
    context : str
        Short description of the input, used in messages.

    Returns
    -------
    (S, N) float array — the validated data (a NaN-cleaned copy if
    ``allow_nan`` and non-finite values were present).
    """
    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        raise ValueError(
            f"The {context} must be 2-D (species x patch); got {data.ndim}-D "
            f"with shape {data.shape}."
        )

    S, N = data.shape

    if S == 0:
        raise ValueError(
            f"The {context} contains no species. Expected a header row of "
            "species names, one patch-label column, and numeric abundances."
        )

    # --- non-finite values -------------------------------------------------
    bad = ~np.isfinite(data)
    if bad.any():
        rows = np.flatnonzero(bad.any(axis=1))
        n_bad = int(bad.sum())
        if allow_nan:
            warnings.warn(
                f"{n_bad} non-finite value(s) in the {context} treated as zero "
                f"(species: {_name_offenders(species, rows)}).",
                stacklevel=2,
            )
            data = np.where(bad, 0.0, data)
        else:
            raise ValueError(
                f"{n_bad} non-finite value(s) (NaN/inf) in the {context}, in "
                f"species: {_name_offenders(species, rows)}. Even a single NaN "
                "corrupts species selection and every fitted parameter. Clean "
                "the data, or pass allow_nan=True (CLI: --allow-nan) to treat "
                "missing cells as absences."
            )

    # --- negative abundances ----------------------------------------------
    neg = data < 0
    if neg.any():
        rows = np.flatnonzero(neg.any(axis=1))
        raise ValueError(
            f"{int(neg.sum())} negative value(s) in the {context}, in species: "
            f"{_name_offenders(species, rows)}. Abundances (stem counts or "
            "crown-area sums) cannot be negative — check for sentinel values "
            "such as -9999 used to mark missing data."
        )

    # --- patch count -------------------------------------------------------
    # Note: S > N is NOT a transposition signal. A species-rich plot can
    # legitimately have more species than patches. Too few *patches*, however,
    # is decisive: no spatial distribution can be estimated from them, and it
    # is also what a transposed matrix looks like.
    if N < int(min_patches):
        raise ValueError(
            f"The {context} has only {N} patch(es) ({S} species). At least "
            f"{min_patches} patches are needed to estimate a spatial "
            "distribution. If your matrix is species-as-rows, transpose it: "
            "the expected CSV layout is one row per patch, one column per "
            "species."
        )
    if N < int(warn_patches):
        warnings.warn(
            f"Only {N} patches in the {context}; fitted parameters will be "
            "poorly determined. Consider a finer spatial scale.",
            stacklevel=2,
        )

    return data
