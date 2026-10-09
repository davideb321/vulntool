"""Golden test: the census path on BCI, a rectangular plot.

Bins the raw 2005 BCI stem list and checks the result against the matrices the
published analysis produced with its own coarse-graining script.

BCI records DBH in millimetres, so ``dbh_scale=1``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vulntool import census_matrices
from vulntool.footprint import normalize_footprint

from conftest import align_to_reference, requires_capsule

EXTENT = (0.0, 1000.0, 0.0, 500.0)
L = 50.0
N_PATCHES = 200  # 20 x 10
N_SPECIES = 299


@pytest.fixture(scope="module")
def bci(capsule):
    stems = pd.read_csv(capsule / "raw_data/BCI/bci5.txt", sep="\t")
    stems = stems[stems.status == "A"]
    stems = stems[np.isfinite(stems.gx) & np.isfinite(stems.gy)]

    fp = normalize_footprint(EXTENT)
    crown, counts, names, binned = census_matrices(
        stems, L, footprint=fp, x="gx", y="gy", species="sp",
        dbh="dbh", dbh_scale=1.0,
    )

    ref_crown = np.loadtxt(capsule / f"raw_data/BCI/bci.crown.area{int(L)}.dbh1.2005.csv",
                           delimiter=",")
    ref_ind = np.loadtxt(capsule / f"raw_data/BCI/bci.ind.area{int(L)}.dbh1.2005.csv",
                         delimiter=",")
    ref_names = [line.strip() for line in
                 open(capsule / "raw_data/BCI/bci.species.dbh1.2005.csv")]
    ref_xy = list(zip(ref_crown[:, 0], ref_crown[:, 1]))
    return dict(crown=crown, counts=counts, names=names, binned=binned,
                ref_crown=ref_crown, ref_ind=ref_ind, ref_names=ref_names, ref_xy=ref_xy)


@requires_capsule
def test_grid_matches_reference(bci):
    assert bci["binned"].grid[:2] == (20, 10)
    assert bci["binned"].n_patches == N_PATCHES
    assert bci["crown"].shape == (N_SPECIES, N_PATCHES)
    # A rectangular plot is entirely sampled: no patch may be masked out.
    assert bci["binned"].mask.all()


@requires_capsule
def test_stem_counts_match_r_exactly(bci):
    """Counts are integers, so this must be exact — no tolerance."""
    mine = align_to_reference(bci["counts"], bci["names"], bci["binned"],
                              bci["ref_xy"], bci["ref_names"])
    assert np.array_equal(mine, bci["ref_ind"][:, 2:].T)


@requires_capsule
def test_crown_metric_matches_r(bci):
    mine = align_to_reference(bci["crown"], bci["names"], bci["binned"],
                              bci["ref_xy"], bci["ref_names"])
    np.testing.assert_allclose(mine, bci["ref_crown"][:, 2:].T, rtol=1e-9, atol=1e-6)


@requires_capsule
def test_dbh_scale_shifts_crown_but_not_counts(bci, capsule):
    """A units mistake is invisible in xs but not in the crown totals."""
    stems = pd.read_csv(capsule / "raw_data/BCI/bci5.txt", sep="\t")
    stems = stems[stems.status == "A"]
    crown_cm, counts_cm, _, _ = census_matrices(
        stems, L, footprint=normalize_footprint(EXTENT), x="gx", y="gy",
        species="sp", dbh="dbh", dbh_scale=10.0,
    )
    assert np.array_equal(counts_cm, bci["counts"])          # selection unaffected
    ratio = crown_cm.sum() / bci["crown"].sum()
    np.testing.assert_allclose(ratio, 10.0 ** (4.0 / 3.0), rtol=1e-9)
