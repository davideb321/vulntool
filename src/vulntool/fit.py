
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import gamma as gamma_dist

# --- base knobset defaults ---
EPS0 = 0.0                 # xs <= EPS0 counts as zero/absent
MIN_POS_POINTS = 5         # min positive patches to attempt a per-species fit
STEP_SIZE = 1.0            # fine scan step in M
COARSE_FACTOR = 10         # coarse step = COARSE_FACTOR * STEP_SIZE
PATIENCE = 50              # number of coarse scan points
REFINE_HALFWIDTH_STEPS = 20  # +/- steps around best coarse M for the fine scan


# ------------------------------------------------------------------
# Empirical CDF helpers
# ------------------------------------------------------------------
def _find_cumulative(data_1d):
    unique_data, counts = np.unique(data_1d, return_counts=True)
    cdf = np.cumsum(counts) / data_1d.size
    return unique_data, cdf


# ------------------------------------------------------------------
# Single-species Gamma fit (positive-only CDF least squares)
# ------------------------------------------------------------------
def fit_single_species_positive_only(xs_s, *, eps0=EPS0, init_params=None):
    """Fit Gamma(shape=delta, rate=beta) to the positive xs of one species.

    Returns ``((beta, delta), gof)`` where ``gof = -SSE`` of the CDF residuals.
    """
    x = np.asarray(xs_s, dtype=float).ravel()
    x = x[np.isfinite(x)]
    pos = x[x > float(eps0)]
    n_pos = int(pos.size)

    if n_pos < 2:
        if init_params is not None:
            init_params = np.asarray(init_params, dtype=float)
            if init_params.shape == (2,) and np.all(np.isfinite(init_params)) and np.all(init_params > 0):
                return init_params.copy(), 0.0
        return np.array([1.0, 1.0], dtype=float), 0.0

    u, F_emp = _find_cumulative(pos)

    def objective(params):
        beta, delta = params
        if beta <= 0 or delta <= 0:
            return np.inf * np.ones_like(F_emp)
        F_th = gamma_dist.cdf(u, delta, scale=1.0 / beta)
        return F_emp - F_th

    m = float(np.mean(pos))
    v = float(np.var(pos))
    if (v <= 0) or (not np.isfinite(v)) or (not np.isfinite(m)) or (m <= 0):
        x0 = np.array([1.0, 1.0], dtype=float)
    else:
        x0 = np.array([m / v, (m * m) / v], dtype=float)

    if init_params is not None:
        init_params = np.asarray(init_params, dtype=float)
        if init_params.shape == (2,) and np.all(np.isfinite(init_params)) and np.all(init_params > 0):
            x0 = init_params

    result = least_squares(objective, x0, bounds=(1e-7, np.inf))
    beta_opt, delta_opt = result.x
    gof = -float(np.sum(result.fun ** 2))
    return np.array([float(beta_opt), float(delta_opt)], dtype=float), gof


# ------------------------------------------------------------------
# All-species fit at a fixed M
# ------------------------------------------------------------------
def fit_at_fixed_M(data, M, *, eps0=EPS0, min_pos_points=MIN_POS_POINTS, init_params=None):
    """Fit every species' Gamma at a fixed scalar (or per-patch) M.

    Returns ``(fitted_params, total_gof)`` with ``fitted_params`` of shape
    (S, 2) holding ``(beta, delta)`` per species.
    """
    data = np.asarray(data, dtype=float)
    S, N = data.shape
    M_arr = float(M) * np.ones(N) if np.isscalar(M) or np.ndim(M) == 0 else np.asarray(M, dtype=float)

    p = data / M_arr.reshape(1, -1)
    free = 1.0 - np.sum(p, axis=0)
    xs_data = p / free.reshape(1, -1)

    fitted_params = np.zeros((S, 2), dtype=float)
    total_gof = 0.0

    for s in range(S):
        init_s = None if init_params is None else init_params[s]
        x_s = xs_data[s]
        n_pos = int(np.sum(np.isfinite(x_s) & (x_s > float(eps0))))

        if n_pos < int(min_pos_points):
            if init_s is not None:
                init_s = np.asarray(init_s, dtype=float)
                if init_s.shape == (2,) and np.all(np.isfinite(init_s)) and np.all(init_s > 0):
                    fitted_params[s] = init_s
                else:
                    fitted_params[s] = np.array([1.0, 1.0])
            else:
                fitted_params[s] = np.array([1.0, 1.0])
            continue

        opt_s, gof_s = fit_single_species_positive_only(x_s, eps0=eps0, init_params=init_s)
        fitted_params[s] = opt_s
        total_gof += float(gof_s)

    return fitted_params, float(total_gof)


def _scan_over_M(data, M_values, *, eps0, min_pos_points, init_params=None):
    """Scan M values, warm-starting each fit from the previous; keep best GOF."""
    best_gof = -np.inf
    best_M = None
    best_params = None
    current_init = init_params

    for M in M_values:
        fit_params, gof = fit_at_fixed_M(
            data, float(M), eps0=eps0, min_pos_points=min_pos_points, init_params=current_init
        )
        current_init = fit_params
        if gof > best_gof:
            best_gof = float(gof)
            best_M = float(M)
            best_params = fit_params

    return best_M, best_gof, best_params


# ------------------------------------------------------------------
# Full homogeneous-M fit: scan M0, fit (beta, delta) per species
# ------------------------------------------------------------------
def fit_homogeneous_M(
    data,
    *,
    eps0=EPS0,
    min_pos_points=MIN_POS_POINTS,
    step_size=STEP_SIZE,
    coarse_factor=COARSE_FACTOR,
    patience=PATIENCE,
    refine_halfwidth_steps=REFINE_HALFWIDTH_STEPS,
    verbose=False,
):
    """Fit the homogeneous-M model to a (S, N) abundance matrix.

    Returns ``(M0, betas, deltas, gof)``:
      M0     : fitted scalar carrying capacity
      betas  : (S,) Gamma rate per species
      deltas : (S,) Gamma shape per species
      gof    : total goodness of fit (-SSE) at the optimum
    """
    data = np.asarray(data, dtype=float)
    patch_totals = data.sum(axis=0)
    min_M = float(np.max(patch_totals) + 1.0)

    fine_step = float(step_size)
    coarse_step = float(coarse_factor) * fine_step

    # Coarse scan
    M_coarse = min_M + coarse_step * np.arange(int(patience), dtype=float)
    best_M, best_gof, best_params = _scan_over_M(
        data, M_coarse, eps0=eps0, min_pos_points=min_pos_points, init_params=None
    )
    if verbose:
        print(f"[coarse] best M0={best_M} gof={best_gof}")

    # Fine scan around the best coarse M
    hw = int(refine_halfwidth_steps)
    M_refine = best_M + fine_step * np.arange(-hw, hw + 1, dtype=float)
    M_refine = M_refine[M_refine >= min_M]
    refine_M, refine_gof, refine_params = _scan_over_M(
        data, M_refine, eps0=eps0, min_pos_points=min_pos_points, init_params=best_params
    )
    if refine_gof > best_gof:
        best_M, best_gof, best_params = refine_M, refine_gof, refine_params
    if verbose:
        print(f"[fine] best M0={best_M} gof={best_gof}")

    betas = best_params[:, 0]
    deltas = best_params[:, 1]
    return best_M, betas, deltas, best_gof
