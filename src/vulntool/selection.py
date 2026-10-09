"""Species selection: a two-step abundance/occupancy filter.

The filter keeps the commonest species and drops those too sparse to fit:

1. Order species by descending total abundance; keep the smallest set whose
   cumulative abundance reaches ``frac_keep`` (default 0.95).
2. From those, drop any species absent from more than ``max_zero_frac`` of
   patches (default 0.90).

Selection is deterministic (ties broken by original species index).
"""
from __future__ import annotations

import numpy as np

FRAC_KEEP = 0.95
MAX_ZERO_FRAC = 0.90


def select_species(data, frac_keep=FRAC_KEEP, max_zero_frac=MAX_ZERO_FRAC):
    """Apply the two-step species filter to a (S, N) abundance matrix.

    Parameters
    ----------
    data : (S, N) array
        Abundance per species (rows) x patch (cols). Selection uses the row
        totals of this matrix.
    frac_keep : float
        Cumulative-abundance fraction to retain (step 1).
    max_zero_frac : float
        Maximum fraction of empty patches allowed (step 2).

    Returns
    -------
    data_sel : (S_sel, N) array, ordered by descending total abundance
    sel_idx  : (S_sel,) original 0-based species indices, aligned to data_sel
    stats    : dict of selection diagnostics
    """
    data = np.asarray(data, dtype=float)
    S, N = data.shape
    totals = data.sum(axis=1)

    idx = np.arange(S)
    # descending totals; ties broken by ascending original index
    order = np.lexsort((idx, -totals))
    data_sorted = data[order]
    totals_sorted = totals[order]

    total_all = totals_sorted.sum()
    # A non-finite total makes `cumfrac` all-NaN, and np.searchsorted on an
    # all-NaN array returns 0 -> k_frac == 1, i.e. every species but one is
    # silently discarded. Callers should have rejected NaN upstream
    # (validate.validate_matrix); this guard makes the collapse impossible.
    if not np.isfinite(total_all) or total_all <= 0:
        stats = {
            "S_total": int(S), "S_after_frac": 0, "S_after_zero": 0,
            "removed_by_frac": int(S), "removed_by_zero": 0,
            "frac_keep": float(frac_keep), "max_zero_frac": float(max_zero_frac),
        }
        return data_sorted[:0], order[:0], stats

    # step 1: cumulative fraction
    cumfrac = np.cumsum(totals_sorted) / total_all
    k_frac = int(np.searchsorted(cumfrac, frac_keep, side="left")) + 1
    k_frac = max(1, min(k_frac, S))

    data_kept = data_sorted[:k_frac]
    idx_kept = order[:k_frac]
    removed_by_frac = int(S - k_frac)

    # step 2: max zero fraction
    zero_frac = (data_kept == 0).sum(axis=1) / data_kept.shape[1]
    ok = zero_frac <= max_zero_frac
    data_sel = data_kept[ok]
    sel_idx = idx_kept[ok]
    removed_by_zero = int(k_frac - data_sel.shape[0])

    stats = {
        "S_total": int(S),
        "S_after_frac": int(k_frac),
        "S_after_zero": int(data_sel.shape[0]),
        "removed_by_frac": int(removed_by_frac),
        "removed_by_zero": int(removed_by_zero),
        "frac_keep": float(frac_keep),
        "max_zero_frac": float(max_zero_frac),
    }
    return data_sel, sel_idx, stats
