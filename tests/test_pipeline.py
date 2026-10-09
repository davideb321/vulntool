"""Golden test: the standalone port must reproduce the paper's numbers.

The oracle is a stored ``base`` fit from the reproducibility capsule
(BCI, crown metric, L=50, frac_keep=0.95, max_zero_frac=0.90). We reproduce it
by fitting the crown matrix while selecting species on the stem-count matrix
(the capsule's policy), and comparing M0, beta, delta, and W_alpha.

Like the other golden tests, this reads the capsule data, which are not
shipped with vulntool. Set ``VULNTOOL_CAPSULE_DATA`` to run it. The
data-independent regression guard is ``test_golden_synthetic.py``, which runs
everywhere.
"""
import json

import numpy as np
import pytest

from vulntool import compute_vulnerability
from vulntool.io import load_bare_matrix

from conftest import requires_capsule

CROWN = "data/BCI/bci.crown.area50.dbh1.2005.csv"
IND = "data/BCI/bci.ind.area50.dbh1.2005.csv"
FIT = ("fit_results/fitting_results_BCI_crown_method-base_L-50"
       "_frac-0.950_zmax-0.900_step-1.0_pat-50.json")

pytestmark = requires_capsule


@pytest.fixture(scope="module")
def golden(capsule):
    with open(capsule / FIT) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def result(capsule):
    crown, _, _ = load_bare_matrix(capsule / CROWN)
    ind, _, _ = load_bare_matrix(capsule / IND)
    # Fit crown, select species on stem counts (paper policy).
    return compute_vulnerability(crown, selection_source=ind)


def test_species_count(result, golden):
    assert len(result.table) == len(golden["selected_species_idx"]) == 101


def test_M0(result, golden):
    stored_M = np.asarray(golden["fit"]["best_M"], dtype=float)
    # capsule stores a per-patch (broadcast) M vector; homogeneous -> all equal
    assert np.allclose(stored_M, stored_M[0])
    assert result.M0 == pytest.approx(float(stored_M[0]), rel=0, abs=1e-6)


def test_betas_deltas(result, golden):
    betas_ref = np.asarray(golden["fit"]["optimal_betas"], dtype=float)
    deltas_ref = np.asarray(golden["fit"]["optimal_deltas"], dtype=float)
    np.testing.assert_allclose(result.table["betabar"].to_numpy(), betas_ref, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(result.table["delta"].to_numpy(), deltas_ref, rtol=1e-6, atol=1e-8)


def test_w_alpha(result, golden):
    W_ref = np.asarray(golden["postproc"]["derived"]["W_alpha"], dtype=float)
    np.testing.assert_allclose(result.table["W_alpha"].to_numpy(), W_ref, rtol=1e-6, atol=1e-8)
