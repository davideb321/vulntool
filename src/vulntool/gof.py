"""How well the model fits — the score behind the choice of patch size.

For each species we compare two curves: how its abundance is actually spread
across patches, and how the fitted model says it should be. The score is the
average squared gap between them, weighted by how many patches contributed::

    misfit = n_present * mean( (F_observed - F_fitted)^2 )

That weighting is what makes the number comparable across patch sizes, and it
brings a natural mark to judge against: were the model exactly right, the score
would average ``1/6``. So below ``1/6`` counts as a good fit, and the tool takes
the smallest patches whose typical species still clears it — the most spatial
detail available before the description starts to break down.

Formally this is a one-sample Cramér-von Mises statistic, computed only over the
patches where the species is actually present: the fitted distribution is
continuous, so it puts no weight on a value of exactly zero, and including
absences would score the model against something it never claimed to describe.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import gamma as gamma_dist

from .theory import get_xs_from_data

#: Reference (null) expectation of the CvM proxy under a correct model.
CVM_REFERENCE = 1.0 / 6.0

#: Minimum positive observations for a species to contribute a CvM value.
MIN_OBS_CVM = 5


def cvm_gamma_xs(xs, betas, deltas, *, min_obs=MIN_OBS_CVM):
    """Per-species CvM proxy for the Gamma/M fit in xs-space (positive xs only).

    Parameters
    ----------
    xs : (S, N) array
        Rescaled abundance per species x patch (see
        :func:`vulntool.theory.get_xs_from_data`).
    betas, deltas : (S,) arrays
        Fitted Gamma rate and shape per species.
    min_obs : int
        Species with fewer than this many positive patches yield ``nan``.

    Returns
    -------
    (S,) array of CvM values (``nan`` where undefined).
    """
    betas = np.asarray(betas, dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    xs = np.asarray(xs, dtype=float)
    S = len(betas)
    cvm = np.full(S, np.nan)
    for s in range(S):
        b, d = float(betas[s]), float(deltas[s])
        if not (b > 0 and d > 0 and np.isfinite(b) and np.isfinite(d)):
            continue
        row = xs[s]
        pos = np.sort(row[np.isfinite(row) & (row > 0)])
        n_pos = len(pos)
        if n_pos < int(min_obs):
            continue
        F_emp = np.arange(1, n_pos + 1) / n_pos
        F_fit = gamma_dist.cdf(pos, d, scale=1.0 / b)
        cvm[s] = n_pos * np.mean((F_emp - F_fit) ** 2)
    return cvm


def median_cvm(data, M, betas, deltas, *, min_obs=MIN_OBS_CVM):
    """Median across species of the Gamma/M CvM proxy for a completed fit.

    Convenience wrapper: rescales ``data`` to xs at the fitted ``M`` and returns
    ``nanmedian`` of :func:`cvm_gamma_xs`. Compared against :data:`CVM_REFERENCE`
    (``1/6``) this is the statistic used to pick a patch size.
    """
    xs, _ = get_xs_from_data(data, M)
    cvm = cvm_gamma_xs(xs, betas, deltas, min_obs=min_obs)
    if np.all(np.isnan(cvm)):
        return float("nan"), cvm
    return float(np.nanmedian(cvm)), cvm
