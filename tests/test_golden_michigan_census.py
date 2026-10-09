"""Golden test: the census path on Michigan, an irregular plot.

The Michigan Big Woods plot is a staircase — a union of five axis-aligned
rectangles covering 23 ha inside a 32 ha bounding box. It is the case that
breaks every rectangle assumption, and the reason the footprint machinery
exists: binning it as its bounding box invents 36 phantom empty patches at
L=50 (128 instead of 92), which corrupts every zero-fraction and hence every
fitted parameter *without raising anything*.

Checked against the original script in which the geometry was hardcoded.

Michigan records DBH in centimetres, so ``dbh_scale=10``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vulntool import census_matrices, default_scales
from vulntool.footprint import grid_shape, normalize_footprint, valid_mask

from conftest import align_to_reference, requires_capsule

BBOX = (-300.0, 500.0, 0.0, 400.0)
#: The five rectangles whose union is the sampled plot (R script, lines 23-29).
FOOTPRINT = [
    (-100.0, 300.0, 0.0, 400.0),
    (-200.0, -100.0, 100.0, 400.0),
    (-300.0, -200.0, 100.0, 200.0),
    (300.0, 400.0, 0.0, 200.0),
    (400.0, 500.0, 0.0, 100.0),
]
#: The same plot spelled the other way round: what is missing from the bbox.
#: Four rectangles instead of five, and exactly equivalent.
EXCLUSIONS = [
    (-300.0, -100.0, 0.0, 100.0),
    (-300.0, -200.0, 200.0, 400.0),
    (300.0, 400.0, 200.0, 400.0),
    (400.0, 500.0, 100.0, 400.0),
]
ALIVE = ("M", "AL", "B", "R")
#: Valid patches per scale, from the R script's own mask files.
EXPECTED_PATCHES = {10: 2300, 20: 575, 25: 368, 50: 92, 100: 23}


@pytest.fixture(scope="module")
def footprint():
    return normalize_footprint(FOOTPRINT, bbox=BBOX)


@pytest.fixture(scope="module")
def excluded_footprint():
    return normalize_footprint(None, bbox=BBOX, holes=EXCLUSIONS)


@pytest.fixture(scope="module")
def stems(capsule):
    df = pd.read_csv(capsule / "raw_data/Michigan/2014census_cortag_gxy.txt", sep="\t")
    df = df[df.codes.isin(ALIVE)]
    return df[np.isfinite(df.gx) & np.isfinite(df.gy)]


@requires_capsule
def test_footprint_area(footprint):
    assert footprint.area() == pytest.approx(230_000.0)
    assert not footprint.is_rectangle


@requires_capsule
def test_scale_ladder_is_derived_from_geometry(footprint):
    """The paper's Michigan scale list should fall out of the footprint alone.

    L must divide the bounding box and land on every footprint edge; with the
    default min_L=10 and min_patches=25 that leaves exactly [10, 20, 25, 50, 100].
    """
    assert default_scales(footprint) == [10.0, 20.0, 25.0, 50.0, 100.0]


@requires_capsule
@pytest.mark.parametrize("L", sorted(EXPECTED_PATCHES))
def test_mask_matches_r(footprint, capsule, L):
    nx, ny, _, _ = grid_shape(footprint, L)
    mask = valid_mask(footprint, nx, ny)
    ref = np.loadtxt(capsule / f"raw_data/Michigan/mich.mask.area{L}.2014.csv",
                     delimiter=",").astype(bool)
    # vulntool walks the grid x-fastest, the R script y-fastest.
    assert np.array_equal(mask.reshape(ny, nx).T.ravel(), ref)
    assert int(mask.sum()) == EXPECTED_PATCHES[L]


@requires_capsule
@pytest.mark.parametrize("L", sorted(EXPECTED_PATCHES))
def test_exclusion_spelling_matches_r_too(excluded_footprint, capsule, L):
    """Describing the plot by what is missing must reach the same R mask."""
    nx, ny, _, _ = grid_shape(excluded_footprint, L)
    mask = valid_mask(excluded_footprint, nx, ny)
    ref = np.loadtxt(capsule / f"raw_data/Michigan/mich.mask.area{L}.2014.csv",
                     delimiter=",").astype(bool)
    assert np.array_equal(mask.reshape(ny, nx).T.ravel(), ref)


@requires_capsule
def test_exclusions_bin_the_census_identically(footprint, excluded_footprint, stems):
    """Equivalence must survive the whole binning pass, not just the mask."""
    kw = dict(x="gx", y="gy", species="spcode", dbh="dbh", dbh_scale=10.0)
    fit_a, sel_a, names_a, binned_a = census_matrices(stems, 50.0,
                                                     footprint=footprint, **kw)
    fit_b, sel_b, names_b, binned_b = census_matrices(stems, 50.0,
                                                     footprint=excluded_footprint, **kw)
    assert names_a == names_b
    assert binned_a.patches == binned_b.patches
    np.testing.assert_array_equal(fit_a, fit_b)
    np.testing.assert_array_equal(sel_a, sel_b)


@requires_capsule
def test_crown_matrix_matches_r(footprint, stems, capsule):
    L = 50
    crown, counts, names, binned = census_matrices(
        stems, L, footprint=footprint, x="gx", y="gy", species="spcode",
        dbh="dbh", dbh_scale=10.0,
    )
    assert binned.n_patches == EXPECTED_PATCHES[L]

    ref = np.loadtxt(capsule / f"raw_data/Michigan/mich.crown.area{L}.dbh1.2014.csv",
                     delimiter=",")
    ref_names = [line.strip() for line in
                 open(capsule / "raw_data/Michigan/mich.species.dbh1.2014.csv")]
    mine = align_to_reference(crown, names, binned,
                              list(zip(ref[:, 0], ref[:, 1])), ref_names)
    np.testing.assert_allclose(mine, ref[:, 2:].T, rtol=1e-9, atol=1e-6)


@requires_capsule
def test_bbox_instead_of_footprint_corrupts_zero_fractions(footprint, stems):
    """Guard the failure this module exists to prevent.

    Using the bounding box keeps the same species but dilutes every
    zero-fraction with patches that are not sampled ground — silently.
    """
    L = 50
    _, counts_fp, _, binned_fp = census_matrices(
        stems, L, footprint=footprint, x="gx", y="gy", species="spcode")
    _, counts_bbox, _, binned_bbox = census_matrices(
        stems, L, footprint=BBOX, x="gx", y="gy", species="spcode")

    assert binned_fp.n_patches == 92
    assert binned_bbox.n_patches == 128

    zf_fp = (counts_fp == 0).mean(axis=1)
    zf_bbox = (counts_bbox == 0).mean(axis=1)
    # Every species looks emptier than it is, by the 36 phantom patches.
    assert np.all(zf_bbox >= zf_fp - 1e-12)
    assert zf_bbox.max() > zf_fp.max()


@requires_capsule
def test_non_dividing_scale_is_rejected(footprint):
    """An irregular footprint cannot use a scale that misaligns the grid."""
    with pytest.raises(ValueError, match="does not divide the plot exactly"):
        grid_shape(footprint, 30.0)
