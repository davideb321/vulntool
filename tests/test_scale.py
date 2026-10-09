"""Scale-selection (census) path: binning, CvM scan, and the census pipeline."""
import numpy as np
import pandas as pd
import pytest

from vulntool import (
    scan_scales,
    compute_vulnerability_census,
    default_scales,
    census_matrices,
    crown_from_dbh,
)


def _make_stems(n=6000, extent=(0.0, 200.0, 0.0, 200.0), n_species=8, seed=0,
                cluster_sd=12.0, parents_per_species=6):
    """A *clustered* synthetic census (Thomas process: parents + offspring).

    Uniformly random stems would be a homogeneous Poisson process — i.e. no
    spatial aggregation at all, which is precisely the phenomenon this package
    measures. Fitting the scale scan against unaggregated data would tell us
    almost nothing, so each species gets its own cluster centres and its stems
    are scattered around them.
    """
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = extent

    # abundance-skewed species
    weights = np.geomspace(1.0, 0.05, n_species)
    weights /= weights.sum()
    counts = rng.multinomial(n, weights)

    xs, ys, sp = [], [], []
    for i, k in enumerate(counts):
        if k == 0:
            continue
        px = rng.uniform(xmin, xmax, parents_per_species)
        py = rng.uniform(ymin, ymax, parents_per_species)
        pick = rng.integers(0, parents_per_species, k)
        # wrap offspring back into the plot so the intensity stays roughly flat
        cx = (px[pick] + rng.normal(0.0, cluster_sd, k) - xmin) % (xmax - xmin) + xmin
        cy = (py[pick] + rng.normal(0.0, cluster_sd, k) - ymin) % (ymax - ymin) + ymin
        xs.append(cx)
        ys.append(cy)
        sp.append(np.full(k, f"Sp{i}"))

    dbh = rng.gamma(shape=2.0, scale=8.0, size=n) + 1.0
    return pd.DataFrame({"x": np.concatenate(xs), "y": np.concatenate(ys),
                         "species": np.concatenate(sp), "dbh": dbh})


def test_crown_from_dbh():
    d = np.array([1.0, 8.0, 27.0])
    np.testing.assert_allclose(crown_from_dbh(d), d ** (4.0 / 3.0))


def test_default_scales_tile_squares_and_are_bounded():
    W, H, min_L, min_patches = 200.0, 100.0, 10.0, 8
    Ls = default_scales((0.0, W, 0.0, H), min_L=min_L, min_patches=min_patches)
    assert Ls  # non-empty
    L_max = (W * H / min_patches) ** 0.5
    for L in Ls:
        # within the [min_L, L_max] band
        assert min_L - 1e-6 <= L <= L_max + 1e-6
        # whole, square patches: L divides both sides
        assert abs(W / L - round(W / L)) < 1e-9
        assert abs(H / L - round(H / L)) < 1e-9
    assert sorted(Ls) == Ls  # ascending, de-duplicated


def test_census_matrices_shapes():
    stems = _make_stems()
    fit, sel, names, binned = census_matrices(
        stems, 50.0, dbh="dbh", extent=(0, 200, 0, 200)
    )
    assert fit.shape == sel.shape
    assert fit.shape[1] == 16  # 200/50 = 4 -> 4x4 patches
    assert binned.n_patches == 16
    assert len(names) == fit.shape[0]
    # crown (fit) totals differ from stem counts (sel); counts are integer
    assert np.allclose(sel.sum(), len(stems))
    assert not np.allclose(fit.sum(), sel.sum())


def test_scan_scales_returns_table():
    stems = _make_stems()
    scan = scan_scales(stems, extent=(0, 200, 0, 200), dbh="dbh",
                       frac_keep=1.0, max_zero_frac=1.0, verbose=False)
    assert set(scan.table.columns) == {"L", "n_patches", "n_species", "median_cvm"}
    assert len(scan.table) >= 2
    assert scan.reference == pytest.approx(1.0 / 6.0)
    # recommended_L, if any, is one of the scanned scales
    if scan.recommended_L is not None:
        assert scan.recommended_L in set(scan.table["L"])
    # results dict holds a VulnerabilityResult per scanned scale
    for L in scan.table["L"]:
        assert L in scan.results


def test_census_pipeline_fixed_L():
    stems = _make_stems()
    result, scan = compute_vulnerability_census(
        stems, L=50.0, extent=(0, 200, 0, 200), dbh="dbh",
        frac_keep=1.0, max_zero_frac=1.0, verbose=False,
    )
    assert scan is None
    assert result.M0 > 0
    assert np.isfinite(result.table["W_alpha"]).all()
    assert "cvm" in result.table.columns


def test_census_pipeline_auto_scale():
    stems = _make_stems()
    result, scan = compute_vulnerability_census(
        stems, extent=(0, 200, 0, 200), dbh="dbh",
        frac_keep=1.0, max_zero_frac=1.0, verbose=False,
    )
    assert scan is not None
    assert result.M0 > 0
    assert np.isfinite(result.cvm)


def test_census_pipeline_with_paper_default_thresholds():
    """Exercise the real selection policy, not a disabled one.

    The other census tests pass frac_keep=1.0/max_zero_frac=1.0, which switches
    the paper's two-step filter off entirely; this one leaves the defaults
    (0.95 / 0.90) in place.
    """
    stems = _make_stems()
    result, scan = compute_vulnerability_census(
        stems, extent=(0, 200, 0, 200), dbh="dbh", verbose=False,
    )
    assert 0 < len(result.table) <= 8
    assert result.selection["frac_keep"] == 0.95
    assert result.selection["max_zero_frac"] == 0.90
    assert np.isfinite(result.table["W_alpha"]).all()
