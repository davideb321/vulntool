"""The maths behind the vulnerability. See the paper for the derivation.

Public symbols
--------------
get_xs_from_data     : abundance relative to the room left in each patch
get_p_mean_from_data : how much of an average patch a species takes up
w_alpha              : the vulnerability index, from the two fitted numbers
compute_derived      : everything derived from a finished fit, in one dict
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.integrate as integrate
from scipy.integrate import IntegrationWarning


# ------------------------------------------------------------------
# Rescaled abundance variable and mean occupancy
# ------------------------------------------------------------------
def get_xs_from_data(data, M):
    """Rescaled abundance ``xs_{s,j} = p_{s,j} / (1 - sum_s p_{s,j})``.

    Parameters
    ----------
    data : (S, N) array
        Raw abundance (stem counts or crown-area sums) per species x patch.
    M : scalar or (N,) array
        Patch carrying capacity. A scalar is the homogeneous-M case.

    Returns
    -------
    xs : (S, N) array
    S : int
    """
    data = np.asarray(data, dtype=float)
    S, N = data.shape
    ps = data / np.asarray(M, dtype=float)
    free_sp = 1.0 - np.sum(ps, axis=0)  # (N,)
    xs = ps / free_sp.reshape(1, N)
    return xs, S


def get_p_mean_from_data(data, M):
    """Mean occupancy per species: ``mean_j(n_{s,j} / M_j)``."""
    data = np.asarray(data, dtype=float)
    ps = data / np.asarray(M, dtype=float)
    return np.mean(ps, axis=1)


# ------------------------------------------------------------------
# Self-consistency integral B_alpha (general, per-species)
# ------------------------------------------------------------------
def _integrand_B(lam, betabars, deltas, s):
    term1 = 1.0 + lam / betabars[s]
    return np.exp(-lam - np.sum(deltas * np.log(1.0 + lam / betabars))) / term1


def _get_B_single(betabarhats, deltas, s):
    # The requested tolerance (1e-16) is tighter than quad can certify, so it
    # reports roundoff on essentially every call. The result is accurate to
    # far better than we need; the warning is pure noise, and emitted once per
    # species it would bury any genuine warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        res, _ = integrate.quad(
            lambda x: _integrand_B(x, betabarhats, deltas, s),
            0, np.inf,
            epsabs=1e-16, epsrel=1e-16, limit=300,
        )
    return res


def get_B_array(betas, deltas):
    """Per-species self-consistency integral ``B_alpha``, one per species.

    ``betas``/``deltas`` are the fitted Gamma rate/shape per species.
    """
    betas = np.asarray(betas, dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    S = len(betas)
    return np.array([_get_B_single(betas, deltas, s) for s in range(S)])


# ------------------------------------------------------------------
# Vulnerability index
# ------------------------------------------------------------------
def _big_delta(betahats, B_00, B_1alpha):
    return (np.mean(betahats * B_1alpha) - B_00 * (1.0 - B_00)) / np.mean(betahats)


def _w_from_B(B_alpha, betahat, which_alive):
    """Vulnerability index from the self-consistency integral B_alpha."""
    S = int(np.sum(which_alive))
    B_00 = np.mean(B_alpha[which_alive])
    B_alpha1 = S * (-B_alpha[which_alive] + B_00) / B_00
    big_delta = _big_delta(betahat[which_alive], B_00, B_alpha1)
    w = np.full(len(B_alpha), np.nan, dtype=np.float64)
    w[which_alive] = (big_delta - B_alpha1) / np.abs(big_delta)
    return w


def w_alpha(betas, deltas, which_alive=None):
    """Spatial-buffering vulnerability index ``W_alpha`` from fitted params.

    Computes the per-species self-consistency integral ``B_alpha`` internally,
    then the index. A more negative W_alpha means the species is *more*
    buffered against competitive exclusion at its abundance; a value near or
    above zero means it is vulnerable.

    Parameters
    ----------
    betas, deltas : (S,) arrays
        Fitted Gamma rate and shape per species.
    which_alive : (S,) bool array, optional
        Subset of species to include; defaults to all.

    Returns
    -------
    (S,) array with W_alpha (NaN for excluded species).
    """
    betas = np.asarray(betas, dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    if which_alive is None:
        which_alive = np.ones(len(betas), dtype=bool)
    else:
        which_alive = np.asarray(which_alive, dtype=bool)

    B_alpha = get_B_array(betas, deltas)
    betahat = B_alpha * deltas
    return _w_from_B(B_alpha, betahat, which_alive)


# ------------------------------------------------------------------
# Bundle of derived quantities from a completed fit
# ------------------------------------------------------------------
def compute_derived(abundance_data, M, betas, deltas):
    """Compute the reported derived quantities from a completed Gamma/M fit.

    Returns a dict with:
      p_alpha  : (S,) mean occupancy
      cv_xs    : scalar, CV of species-mean xs (the spatial-variability signal)
      B_alpha  : (S,) self-consistency integral
      betahat  : (S,) = B_alpha * delta
      W_alpha  : (S,) vulnerability index
      Delta    : scalar community self-consistency residual
    """
    abundance_data = np.asarray(abundance_data, dtype=float)
    betas = np.asarray(betas, dtype=float)
    deltas = np.asarray(deltas, dtype=float)

    xs_data, _ = get_xs_from_data(abundance_data, M)
    x_alpha = np.mean(xs_data, axis=1)
    cv_xs = float(np.sqrt(np.var(x_alpha)) / np.mean(x_alpha))

    B_alpha = get_B_array(betas, deltas)
    betahat = B_alpha * deltas
    p_alpha = get_p_mean_from_data(abundance_data, M)

    which_alive = np.ones(len(betas), dtype=bool)
    W = _w_from_B(B_alpha, betahat, which_alive)

    B_00 = float(np.mean(B_alpha))
    S = abundance_data.shape[0]
    B_alpha1 = S * (-B_alpha + B_00) / B_00
    Delta = float(_big_delta(betahat, B_00, B_alpha1))

    return {
        "p_alpha": p_alpha,
        "cv_xs": cv_xs,
        "B_alpha": B_alpha,
        "betahat": betahat,
        "W_alpha": W,
        "Delta": Delta,
    }
