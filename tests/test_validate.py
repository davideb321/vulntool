"""Malformed input must fail loudly, not produce a confident wrong answer.

Every case here was found by probing the pre-hardening package: all of them
previously ran to completion and returned plausible-looking numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vulntool import compute_vulnerability, compute_vulnerability_census
from vulntool.io import bin_census, crown_from_dbh
from vulntool.selection import select_species

rng = np.random.default_rng(0)


def good_matrix(S=6, N=40):
    """A well-behaved matrix in the canonical patch-rows CSV layout."""
    d = rng.poisson(6, size=(N, S)).astype(float)
    df = pd.DataFrame(d, columns=[f"sp{i}" for i in range(S)])
    df.insert(0, "patch", [f"p{j}" for j in range(N)])
    return df


def stem_list(n=2000, extent=(0, 1000, 0, 500), n_species=6):
    xmin, xmax, ymin, ymax = extent
    return pd.DataFrame({
        "x": rng.uniform(xmin, xmax, n),
        "y": rng.uniform(ymin, ymax, n),
        "species": rng.choice([f"s{i}" for i in range(n_species)], n),
        "dbh": rng.uniform(10, 300, n),
    })


# ---------------------------------------------------------------- NaN

def test_single_nan_is_rejected():
    df = good_matrix()
    df.loc[3, "sp2"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        compute_vulnerability(df)


def test_nan_message_names_the_species():
    df = good_matrix()
    df.loc[3, "sp2"] = np.nan
    with pytest.raises(ValueError, match="sp2"):
        compute_vulnerability(df)


def test_allow_nan_treats_missing_as_absence():
    df = good_matrix()
    df.loc[3, "sp2"] = np.nan
    with pytest.warns(UserWarning, match="treated as zero"):
        res = compute_vulnerability(df, allow_nan=True)
    assert len(res.table) == 6
    assert np.isfinite(res.table["W_alpha"]).all()


def test_nan_no_longer_collapses_selection():
    """The regression this whole module exists for.

    A single NaN used to make cumfrac all-NaN, so searchsorted returned 0 and
    every species but one was silently dropped.
    """
    d = rng.poisson(6, size=(6, 40)).astype(float)
    clean = select_species(d)[2]
    d[2, 3] = np.nan
    stats = select_species(d)[2]
    assert clean["S_after_frac"] == 6
    # Degenerate input must not masquerade as a confident 1-species answer.
    assert stats["S_after_frac"] in (0, 6)


# ---------------------------------------------------------------- other corruption

def test_negative_abundance_is_rejected():
    df = good_matrix()
    df.loc[5, "sp1"] = -20.0
    with pytest.raises(ValueError, match="negative"):
        compute_vulnerability(df)


def test_duplicate_species_names_are_rejected():
    df = good_matrix(S=4)
    df.columns = ["patch", "sp0", "sp1", "sp1", "sp3"]
    with pytest.raises(ValueError, match="Duplicate species name"):
        compute_vulnerability(df)


def test_single_patch_is_rejected():
    with pytest.raises(ValueError, match="At least 4 patches"):
        compute_vulnerability(good_matrix(N=1))


def test_transposed_matrix_warns_via_patch_count():
    """Species-as-rows collapses the patch count, which is the usable signal.

    S > N is *not* usable as a transposition test — Pasoh legitimately has 396
    species over 200 patches — so the only honest signal is that transposing
    leaves implausibly few patches. Between 4 and 10 patches that is a warning
    rather than an error, because a genuinely tiny plot is possible.
    """
    df = good_matrix(S=6, N=40)
    tp = df.set_index("patch").T.reset_index().rename(columns={"index": "species"})
    with pytest.warns(UserWarning, match="Only 6 patches"):
        compute_vulnerability(tp)


def test_badly_transposed_matrix_is_rejected():
    """With few enough species, transposing drops below the hard floor."""
    df = good_matrix(S=3, N=40)
    tp = df.set_index("patch").T.reset_index().rename(columns={"index": "species"})
    with pytest.raises(ValueError, match="transpose"):
        compute_vulnerability(tp)


def test_no_species_columns():
    with pytest.raises(ValueError, match="no species"):
        compute_vulnerability(pd.DataFrame({"patch": ["p0", "p1", "p2", "p3", "p4"]}))


def test_numeric_patch_column_warns():
    """All-numeric input means the first species is about to be eaten silently."""
    df = good_matrix().drop(columns=["patch"])
    with pytest.warns(UserWarning, match="numeric"):
        compute_vulnerability(df)


# ---------------------------------------------------------------- census path

def test_all_stems_outside_extent_reports_the_ranges():
    stems = stem_list(extent=(2000, 3000, 2000, 3000))
    with pytest.raises(ValueError, match="outside the plot extent"):
        compute_vulnerability_census(stems, extent=(0, 1000, 0, 500), L=100, verbose=False)


def test_extent_in_wrong_order_is_rejected():
    """(xmin, ymin, xmax, ymax) is a natural mistake and must not be silent."""
    with pytest.raises(ValueError, match="non-positive|positive width"):
        compute_vulnerability_census(stem_list(), extent=(0, 0, 1000, 500),
                                     L=100, verbose=False)


def test_negative_dbh_names_the_cause():
    with pytest.raises(ValueError, match="negative DBH"):
        crown_from_dbh([10.0, -5.0, 20.0])


def test_missing_column_lists_the_available_ones():
    stems = stem_list().rename(columns={"species": "sp"})
    with pytest.raises(ValueError, match="not found in the stem list"):
        bin_census(stems, 100, extent=(0, 1000, 0, 500))


def test_mask_of_wrong_size_is_rejected():
    from vulntool.footprint import normalize_footprint, valid_mask
    fp = normalize_footprint(None, bbox=(0, 100, 0, 100), mask=np.ones(4, dtype=bool))
    with pytest.raises(ValueError, match="tied to one spatial scale"):
        valid_mask(fp, 5, 5)
