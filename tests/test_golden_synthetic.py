"""Numerical regression guard for the fitting core.

``test_pipeline.py`` checks the port against the paper's own stored fit.
This test pins the same pipeline on the synthetic demo census instead, so that an
accidental change to the M scan, the Gamma fit or the self-consistency integral
fails somewhere rather than nowhere.

**These numbers are a characterisation, not a truth.** They were produced by the
code they now guard, and they say nothing about whether the method is right —
only that it has not silently changed. If a deliberate change moves them, read
``test_pipeline.py``'s verdict first, then re-pin here.
"""
from __future__ import annotations

import numpy as np
import pytest

from vulntool import DEMO_PLOT, compute_vulnerability_census, load_demo_census
from vulntool.io import keep_rows

@pytest.fixture(scope="module")
def result():
    stems = keep_rows(load_demo_census(), "status", ["A"])
    res, scan = compute_vulnerability_census(
        stems, L=25.0, footprint=DEMO_PLOT.footprint,
        dbh="dbh", dbh_units="mm", verbose=False,
    )
    assert scan is None  # an explicit L must not trigger a scan
    return res


def test_shape_and_units(result):
    assert len(result.table) == 25
    assert list(result.table.columns) == [
        "species", "abundance", "p_alpha", "betabar", "delta", "W_alpha", "cvm",
    ]
    assert result.metric == "crown"
    assert result.units == "mm^(4/3)"


def test_pinned_values(result):
    assert result.M0 == pytest.approx(82625.46671, rel=1e-9)
    assert result.cvm == pytest.approx(0.07987470747, rel=1e-6)
    assert result.cv_xs == pytest.approx(1.627309551, rel=1e-6)
    assert result.Delta == pytest.approx(-1.117343672, rel=1e-6)

    t = result.table.set_index("species")
    # a common, a middling and a rare species
    for name, p_alpha, delta, W in [
        ("Alseis", 0.05755136735, 0.5818526499, -2.908069185),
        ("Beilschmiedia", 0.02463309173, 0.4063865439, -2.875546879),
        ("Sloanea", 0.001477236331, 0.7534887541, -0.116962829),
    ]:
        assert t.loc[name, "p_alpha"] == pytest.approx(p_alpha, rel=1e-6)
        assert t.loc[name, "delta"] == pytest.approx(delta, rel=1e-6)
        assert t.loc[name, "W_alpha"] == pytest.approx(W, rel=1e-6)


def test_w_alpha_behaves_as_it_does_on_real_data(result):
    """Sanity independent of the pinned digits, calibrated against real data."""
    t = result.table
    assert (t.W_alpha < 0).all()
    r = np.corrcoef(np.log(t.p_alpha), t.W_alpha)[0, 1]
    assert -0.9 < r < -0.4
