"""Footprint geometry, including exclusion rectangles ("holes").

Pure geometry — no capsule data, no matplotlib, no streamlit — so this runs
everywhere. The anchor is :func:`test_michigan_as_holes_equals_michigan_as_rects`:
the same plot spelled two ways must produce the same patches, cell for cell.
"""
from __future__ import annotations

import numpy as np
import pytest

from vulntool.footprint import (
    Footprint,
    admissible_scales,
    check_scale,
    grid_shape,
    inside_footprint,
    normalize_footprint,
    valid_mask,
)

# The Michigan Big Woods plot, both ways round.
MICHIGAN_BBOX = (-300.0, 500.0, 0.0, 400.0)
MICHIGAN_RECTS = [
    (-100.0, 300.0, 0.0, 400.0),
    (-200.0, -100.0, 100.0, 400.0),
    (-300.0, -200.0, 100.0, 200.0),
    (300.0, 400.0, 0.0, 200.0),
    (400.0, 500.0, 0.0, 100.0),
]
#: The complement of MICHIGAN_RECTS inside MICHIGAN_BBOX.
MICHIGAN_HOLES = [
    (-300.0, -100.0, 0.0, 100.0),
    (-300.0, -200.0, 200.0, 400.0),
    (300.0, 400.0, 200.0, 400.0),
    (400.0, 500.0, 100.0, 400.0),
]
#: Sampled patches per scale, from the paper's R scripts.
MICHIGAN_PATCHES = {10.0: 2300, 20.0: 575, 25.0: 368, 50.0: 92, 100.0: 23}


def _square_with_hole():
    """A 100 x 100 plot with a 20 x 20 clearing in the middle."""
    return normalize_footprint(None, bbox=(0.0, 100.0, 0.0, 100.0),
                               holes=[(40.0, 60.0, 40.0, 60.0)])


# ---------------------------------------------------------------- basic shape
def test_hole_removes_its_area():
    assert _square_with_hole().area() == pytest.approx(100 * 100 - 20 * 20)


def test_overlapping_holes_are_counted_once():
    fp = normalize_footprint(None, bbox=(0.0, 100.0, 0.0, 100.0),
                             holes=[(0.0, 50.0, 0.0, 50.0), (25.0, 75.0, 0.0, 50.0)])
    # Union of the two holes is 75 x 50, not 50x50 + 50x50.
    assert fp.area() == pytest.approx(10000 - 75 * 50)


def test_hole_sticking_out_of_the_bbox_is_clipped():
    fp = normalize_footprint(None, bbox=(0.0, 100.0, 0.0, 100.0),
                             holes=[(-50.0, 50.0, -50.0, 50.0)])
    assert fp.area() == pytest.approx(10000 - 50 * 50)


def test_holes_make_the_footprint_shaped():
    fp = _square_with_hole()
    assert not fp.is_rectangle
    assert fp.has_shape
    assert normalize_footprint((0.0, 100.0, 0.0, 100.0)).is_rectangle


def test_hole_wins_over_an_overlapping_rect():
    fp = normalize_footprint([(0.0, 100.0, 0.0, 100.0)], holes=[(0.0, 50.0, 0.0, 100.0)])
    assert fp.area() == pytest.approx(5000)
    assert not inside_footprint(fp, 25.0, 50.0)
    assert inside_footprint(fp, 75.0, 50.0)


# ---------------------------------------------------------------- patch masks
def test_valid_mask_drops_the_hole_patches():
    fp = _square_with_hole()
    nx, ny, _, _ = grid_shape(fp, 10.0)
    mask = valid_mask(fp, nx, ny)
    assert mask.sum() == 100 - 4          # the 2x2 block of cells inside the hole
    assert fp.area() / 100.0 == mask.sum()  # area and patch count agree


def test_half_open_edges_apply_to_holes_too():
    """A point on a hole's low edge is inside the hole; on its high edge, out."""
    fp = _square_with_hole()
    xs = [39.999, 40.0, 59.999, 60.0]
    got = inside_footprint(fp, xs, [50.0] * 4)
    np.testing.assert_array_equal(got, [True, False, False, True])


# -------------------------------------------------- regressions: "rects is None"
def test_grid_shape_defaults_to_exact_tiling_when_only_holes_are_given():
    """A holes-only footprint has ``rects is None`` but is not a rectangle.

    Without the ``has_shape`` test it would silently get loose ``round`` tiling,
    putting grid lines through the hole edges — so which patches are dropped
    would depend on how W/L happened to round.
    """
    with pytest.raises(ValueError, match="does not divide the plot exactly"):
        grid_shape(_square_with_hole(), 30.0)


def test_admissible_scales_land_on_hole_edges():
    """Hole edges must join the lattice `_edge_gcd` derives the ladder from."""
    holed = normalize_footprint(None, bbox=(0.0, 800.0, 0.0, 400.0),
                                holes=[(300.0, 400.0, 200.0, 400.0)])
    scales = admissible_scales(holed)
    assert scales and all(100.0 % L == 0 for L in scales)
    # The same box with no hole is free to use the geometric ladder, which
    # offers scales that would cut the hole in half.
    plain = admissible_scales(normalize_footprint((0.0, 800.0, 0.0, 400.0)))
    assert plain != scales
    assert any(100.0 % L != 0 for L in plain)


def _square_tiling(fp, L):
    """Both sides an exact multiple of L: whole, square patches."""
    return all(abs(side / L - round(side / L)) < 1e-3 for side in (fp.width, fp.height))


def test_rectangle_ladder_is_whole_metre_divisors():
    """Regression: BCI (1000 x 500) was offered 45.45 and 66.67 but not 50."""
    bci = normalize_footprint((0.0, 1000.0, 0.0, 500.0))
    assert admissible_scales(bci) == [10.0, 20.0, 25.0, 50.0, 100.0, 125.0]


def test_rectangle_ladder_tiles_into_squares_and_is_thinned():
    big = normalize_footprint((0.0, 2000.0, 0.0, 2000.0))
    scales = admissible_scales(big, max_scales=8)
    assert len(scales) == 8 and scales[0] == 10.0 and scales[-1] == 400.0
    assert all(L == int(L) and _square_tiling(big, L) for L in scales)


def test_rectangle_with_few_whole_divisors_gets_exact_fractions():
    sq = normalize_footprint((0.0, 100.0, 0.0, 100.0))
    scales = admissible_scales(sq)          # whole sizes in range: 10, 20 only
    assert 16.6667 in scales and len(scales) > 2
    assert all(_square_tiling(sq, L) for L in scales)


def test_rectangle_with_fractional_sides_falls_back_to_snapped_ladder():
    odd = normalize_footprint((0.0, 1000.5, 0.0, 500.0))
    scales = admissible_scales(odd)
    assert scales and all(abs(1000.5 / L - round(1000.5 / L)) < 1e-3 for L in scales)


def test_check_scale_accepts_sizes_that_tile_the_rectangle():
    bci = normalize_footprint((0.0, 1000.0, 0.0, 500.0))
    assert check_scale(bci, 50) == 50.0
    assert check_scale(bci, "31.25") == 31.25      # the paper's "L=31"
    assert check_scale(bci, 16.67) == 16.6667      # typed for 1000/60
    assert check_scale(bci, 5) == 5.0              # below min_L is allowed


@pytest.mark.parametrize("L, match", [
    (33, "does not divide the plot evenly"),
    (40, "does not divide the plot evenly"),     # divides 1000, not 500
    (2000, "larger than the plot"),
    (250, "only 8 patches"),
    (0, "positive"),
    ("abc", "number of metres"),
])
def test_check_scale_rejects_and_says_why(L, match):
    bci = normalize_footprint((0.0, 1000.0, 0.0, 500.0))
    with pytest.raises(ValueError, match=match):
        check_scale(bci, L)


def test_check_scale_needs_the_grid_on_the_outline_edges():
    holed = normalize_footprint(None, bbox=(0.0, 800.0, 0.0, 400.0),
                                holes=[(300.0, 400.0, 200.0, 400.0)])
    assert check_scale(holed, 50) == 50.0
    with pytest.raises(ValueError, match="not the outline"):
        check_scale(holed, 80)                   # divides 800 x 400, cuts the hole
    # Every size the ladder offers passes, typed back in.
    mich = normalize_footprint(MICHIGAN_RECTS, bbox=MICHIGAN_BBOX)
    for L in admissible_scales(mich):
        assert check_scale(mich, L) == L


# ------------------------------------------------------------------- the anchor
def test_michigan_as_holes_equals_michigan_as_rects():
    a = normalize_footprint(MICHIGAN_RECTS, bbox=MICHIGAN_BBOX)
    b = normalize_footprint(None, bbox=MICHIGAN_BBOX, holes=MICHIGAN_HOLES)

    assert a.area() == b.area() == 230_000.0
    assert admissible_scales(a) == admissible_scales(b) == [10.0, 20.0, 25.0, 50.0, 100.0]

    for L in admissible_scales(a):
        assert grid_shape(a, L) == grid_shape(b, L)
        nx, ny, _, _ = grid_shape(a, L)
        mask_a, mask_b = valid_mask(a, nx, ny), valid_mask(b, nx, ny)
        np.testing.assert_array_equal(mask_a, mask_b)
        assert mask_a.sum() == MICHIGAN_PATCHES[L]


def test_michigan_spellings_agree_on_random_points():
    a = normalize_footprint(MICHIGAN_RECTS, bbox=MICHIGAN_BBOX)
    b = normalize_footprint(None, bbox=MICHIGAN_BBOX, holes=MICHIGAN_HOLES)
    rng = np.random.default_rng(0)
    x = rng.uniform(-300, 500, 5000)
    y = rng.uniform(0, 400, 5000)
    np.testing.assert_array_equal(inside_footprint(a, x, y), inside_footprint(b, x, y))


# ------------------------------------------------------------------ bad input
def test_holes_without_a_bbox_are_rejected():
    with pytest.raises(ValueError, match="bounding box itself is still needed"):
        normalize_footprint(None, holes=[(0.0, 1.0, 0.0, 1.0)])


def test_mask_plus_holes_is_rejected():
    with pytest.raises(ValueError, match="applied twice"):
        normalize_footprint(None, bbox=(0.0, 10.0, 0.0, 10.0),
                            mask=np.ones(4, dtype=bool), holes=[(0.0, 5.0, 0.0, 5.0)])


def test_hole_outside_the_bbox_is_rejected():
    with pytest.raises(ValueError, match="removes nothing"):
        normalize_footprint(None, bbox=(0.0, 100.0, 0.0, 100.0),
                            holes=[(200.0, 300.0, 0.0, 100.0)])


def test_hole_in_the_wrong_coordinate_order_is_rejected():
    # (xmin, ymin, xmax, ymax) instead of (xmin, xmax, ymin, ymax).
    with pytest.raises(ValueError, match="non-positive"):
        normalize_footprint(None, bbox=(0.0, 100.0, 0.0, 100.0),
                            holes=[(40.0, 40.0, 60.0, 60.0)])


def test_normalize_is_idempotent_on_a_holed_footprint():
    fp = _square_with_hole()
    assert normalize_footprint(fp) is fp
    with pytest.raises(ValueError, match="already carries"):
        normalize_footprint(fp, holes=[(0.0, 10.0, 0.0, 10.0)])


def test_holes_can_be_attached_to_an_existing_footprint():
    plain = normalize_footprint((0.0, 100.0, 0.0, 100.0))
    holed = normalize_footprint(plain, holes=[(40.0, 60.0, 40.0, 60.0)])
    assert holed.area() == pytest.approx(9600)
    assert plain.is_rectangle  # frozen dataclass: the original is untouched


def test_positional_construction_still_works():
    """`holes` is the last field so old positional calls keep their meaning."""
    fp = Footprint(MICHIGAN_BBOX, tuple(MICHIGAN_RECTS), None)
    assert fp.rects == tuple(MICHIGAN_RECTS) and fp.holes is None
