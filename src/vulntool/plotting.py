"""The charts: vulnerability against abundance, fit quality by patch size, and
the fit for each species.

matplotlib is an optional dependency (``pip install vulntool[plot]``).
"""
from __future__ import annotations

import numpy as np

#: Shared by every vulntool figure: quiet tick marks, dark-grey tick labels.
TICK_COLOR = "#8a8a8a"
TICK_LABEL_COLOR = "#444444"


def tidy_axes(ax, *, spines=("left", "bottom")):
    """House style: only the listed spines, short grey ticks."""
    for name, spine in ax.spines.items():
        spine.set_visible(name in spines)
        spine.set_color(TICK_COLOR)
    ax.tick_params(which="both", color=TICK_COLOR, labelcolor=TICK_LABEL_COLOR)
    ax.tick_params(which="major", length=3)
    ax.tick_params(which="minor", length=1.8)


def tidy_colorbar(cbar):
    """No box around the bar; ticks styled as on the main axes."""
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(which="both", color=TICK_COLOR,
                        labelcolor=TICK_LABEL_COLOR)
    cbar.ax.tick_params(which="major", length=3)
    cbar.ax.tick_params(which="minor", length=1.8)


def plot_vulnerability(result, *, ax=None, annotate=0, title=None):
    """Scatter each species' vulnerability against how common it is.

    Parameters
    ----------
    result : VulnerabilityResult
    ax : matplotlib Axes, optional
    annotate : int
        Label this many of the most vulnerable species (highest ``W_alpha``)
        with their row number in ``result.table``. Pass ``len(result.table)``
        to label every species.
    title : str, optional
        No title is drawn unless one is given.

    Returns
    -------
    matplotlib Axes
    """
    try:
        import matplotlib.patheffects as pe
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Plotting requires matplotlib. Install it with: pip install vulntool[plot]"
        ) from e

    df = result.table
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))

    top_idx = set()
    if annotate > 0:
        top_idx = set(df.sort_values("W_alpha", ascending=False).head(int(annotate)).index)
    # Labelled species get a marker sized to fit their row number; unlabelled
    # ones stay small dots.
    sizes = [110 if i in top_idx else 30 for i in df.index]

    ax.scatter(df["p_alpha"], df["W_alpha"], c=df["W_alpha"],
                    cmap="coolwarm", s=sizes, edgecolor="k", linewidth=0.3,
                    zorder=2)
    ax.set_xscale("log")
    # p_alpha can span anywhere from under one decade to eight (rare
    # crown-metric species sit down at 1e-8). Whole-decade ticks alone leave
    # a narrow range with one label or none, and matplotlib's LogLocator then
    # falls back to linear ticks; its formatter also hides non-decade labels
    # on a judgement call we can't rely on. So the ticks are placed by hand:
    # the sparsest of 1x / 1,5x / 1,2,5x / every-integer labels that puts at
    # least two in view (every other decade if whole decades crowd), with
    # the remaining 1..9x as unlabelled minor ticks.
    ax.xaxis.set_major_locator(mticker.FixedLocator(_log_ticks(ax.get_xlim())))
    ax.xaxis.set_minor_locator(mticker.FixedLocator(
        _log_ticks(ax.get_xlim(), minor=True)))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    def _log_tick_label(x, _pos):
        if x <= 0:
            return ""
        exp = np.floor(np.log10(x) + 1e-9)
        mantissa = round(x / 10**exp)
        base = rf"10^{{{int(exp)}}}"
        return f"${mantissa}{{\\times}}{base}$" if mantissa != 1 else f"${base}$"

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_log_tick_label))
    ax.axhline(0.0, color="0.4", lw=0.8, ls="--")
    ax.set_xlabel(r"Mean relative abundance $p_\alpha$")
    ax.set_ylabel(r"Vulnerability $W_\alpha$")
    if title:
        ax.set_title(title)
    tidy_axes(ax)

    if top_idx:
        # The row number matches the per-species results table, so it is a
        # single label short enough to sit inside the marker rather than
        # beside it, where names would overlap for nearby points.
        for i in top_idx:
            row = df.loc[i]
            ax.annotate(str(i), (row["p_alpha"], row["W_alpha"]),
                        ha="center", va="center", fontsize=5.5, fontweight="bold",
                        color="black", zorder=3,
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # No colorbar: the colour repeats the y axis, which already gives the
    # value, so it is there only to make the most vulnerable stand out.
    ax.figure.tight_layout()
    return ax


def plot_gof_vs_L(scan, *, ax=None, title=None):
    """How well the model fits at each patch size, and which size was chosen.

    The curve falls as patches get larger and hold more trees. Where it drops
    below the reference line the fit counts as good, and the tool takes the
    smallest such size — the most spatial detail that still fits.

    Parameters
    ----------
    scan : ScaleScan
        Output of :func:`vulntool.scan_scales`.
    ax : matplotlib Axes, optional
    title : str, optional
        No title is drawn unless one is given.

    Returns
    -------
    matplotlib Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Plotting requires matplotlib. Install it with: pip install vulntool[plot]"
        ) from e

    df = scan.table
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot(df["L"], df["median_cvm"], "o-", color="#1a5276", lw=1.6, zorder=3)
    ax.axhline(scan.reference, color="0.4", lw=1.0, ls="--",
               label=f"Reference ({_fraction(scan.reference)})")
    if scan.recommended_L is not None:
        ax.axvline(scan.recommended_L, color="#1e8449", lw=1.2, ls=":",
                   label=f"Selected $L$ = {scan.recommended_L:g} m")
    ax.set_xlabel(r"Patch side length $L$ (m)")
    ax.set_ylabel("Median Cramér–von Mises statistic")
    if title:
        ax.set_title(title)
    tidy_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    ax.figure.tight_layout()
    return ax


#: Species fits: histogram fill and edge, fitted density.
FIT_FILL = "#a9cce3"
FIT_LINE = "#1a5276"


def species_fit_order(result, *, worst_first=False):
    """Row labels of ``result.table`` in the order to show their fits.

    By default the table's own order, i.e. descending abundance, as in the
    paper's supplement. ``worst_first`` orders by descending ``cvm`` instead;
    species without a ``cvm`` value are placed last.
    """
    df = result.table
    if not worst_first:
        return list(df.index)
    return list(df["cvm"].sort_values(ascending=False, na_position="last").index)


def plot_species_fits(result, rows=None, *, ncols=5, panel_size=(2.4, 1.9),
                      legend=True):
    """Empirical distribution of each species against its fitted Gamma density.

    Each panel shows the histogram of the vacancy-adjusted abundance over the patches
    where the species is present, which are the values the Gamma is fitted to,
    together with the fitted density. Patches where the species is absent are
    excluded from the fit; their fraction is reported as text rather than
    drawn. Each panel also reports the fitted parameters ``delta`` and
    ``betabar`` and the Cramér-von Mises statistic ``cvm``, whose expected
    value under a correctly specified model is approximately ``1/6``.

    Parameters
    ----------
    result : VulnerabilityResult
        Must carry ``xs`` (every result from :func:`compute_vulnerability`
        does).
    rows : sequence, optional
        Row labels of ``result.table`` to draw, in this order; default all of
        them, most abundant first. See :func:`species_fit_order`.
    ncols : int
        Panels per row.
    panel_size : (float, float)
        Width and height of one panel, in inches.
    legend : bool
        Add a one-line key under the panels.

    Returns
    -------
    matplotlib Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Plotting requires matplotlib. Install it with: pip install vulntool[plot]"
        ) from e
    if getattr(result, "xs", None) is None:
        raise ValueError(
            "This result does not carry the fitted data (result.xs), so its "
            "fits cannot be drawn. Recompute it with compute_vulnerability."
        )
    df = result.table
    rows = list(df.index) if rows is None else list(rows)
    if not rows:
        raise ValueError("No species to draw.")
    positions = {label: k for k, label in enumerate(df.index)}
    xs = np.asarray(result.xs, dtype=float)

    ncols = max(1, min(int(ncols), len(rows)))
    nrows = -(-len(rows) // ncols)
    key_height = 0.35 if legend else 0.0
    fig, axes = plt.subplots(
        nrows, ncols, squeeze=False,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows + key_height),
    )
    for ax, label in zip(axes.flat, rows):
        row = df.loc[label]
        _species_fit_panel(ax, xs[positions[label]], float(row["delta"]),
                           float(row["betabar"]), float(row["cvm"]),
                           title=f"{label}  {row['species']}")
    for ax in axes.flat[len(rows):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(r"vacancy-adjusted abundance $x_\alpha$", fontsize=7,
                      color=TICK_LABEL_COLOR)

    bottom = 0.0
    if legend:
        bottom = key_height / fig.get_figheight()
        fig.text(
            0.5, bottom * 0.35,
            "Histogram: empirical vacancy-adjusted abundance.  "
            "Line: theoretical fits.",
            ha="center", va="center", fontsize=7.5, color=TICK_LABEL_COLOR,
        )
    fig.tight_layout(rect=(0, bottom, 1, 1), h_pad=0.6, w_pad=0.4)
    return fig


def save_species_fits_pdf(result, path, *, worst_first=False, per_page=25,
                          ncols=5):
    """Every species' fit, ``per_page`` panels to a page, as one PDF.

    Returns ``path``. ``path`` may also be a writable binary file object.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    order = species_fit_order(result, worst_first=worst_first)
    with PdfPages(path) as pdf:
        for start in range(0, len(order), per_page):
            fig = plot_species_fits(result, order[start:start + per_page],
                                    ncols=ncols)
            pdf.savefig(fig)
            plt.close(fig)
    return path


def _species_fit_panel(ax, xs_row, delta, betabar, cvm, *, title):
    """Histogram of the positive values with the fitted Gamma density over it."""
    import matplotlib.ticker as mticker
    from scipy.stats import gamma as gamma_dist

    row = xs_row[np.isfinite(xs_row)]
    pos = row[row > 0]
    n_zero = int(np.sum(row == 0))
    ok = np.isfinite(delta) and np.isfinite(betabar) and delta > 0 and betabar > 0
    scale = 1.0 / betabar if ok else 1.0

    if pos.size:
        # Range: most of the data and most of the fitted mass, so neither a
        # long data tail nor a long model tail squeezes the rest into a corner.
        hi = float(np.quantile(pos, 0.98))
        if ok:
            hi = max(hi, float(gamma_dist.ppf(0.99, delta, scale=scale)))
        hi = hi if hi > 0 else float(pos.max())
        nbins = int(np.clip(round(1.5 * np.sqrt(pos.size)), 10, 20))
        edges = np.linspace(0.0, hi, nbins + 1)
        counts, _ = np.histogram(pos, bins=edges)
        # Normalised by every positive value, including any past the right
        # edge, so the bars are the true density and comparable to the line.
        density = counts / (pos.size * np.diff(edges))
        ax.bar(edges[:-1], density, width=np.diff(edges), align="edge",
               color=FIT_FILL, edgecolor=FIT_LINE, linewidth=0.3)
        top = float(density.max())
        if ok:
            xg = np.linspace(hi / 400, hi, 400)
            yg = gamma_dist.pdf(xg, delta, scale=scale)
            ax.plot(xg, yg, color=FIT_LINE, lw=1.4)
            # A shape below 1 makes the density blow up at zero; judge the
            # height by the bins instead, as the bars do.
            centres = 0.5 * (edges[:-1] + edges[1:])
            top = max(top, float(gamma_dist.pdf(centres, delta, scale=scale).max()))
        ax.set_xlim(0, hi)
        ax.set_ylim(0, 1.7 * top if top > 0 else 1)

    ax.set_title(_shorten(title, 28), fontsize=8, loc="left", pad=3,
                 color="black")
    lines = [rf"$\delta$ = {delta:.3g}", rf"$\bar\beta$ = {betabar:.3g}",
             f"cvm = {cvm:.2g}" if np.isfinite(cvm) else "cvm = n/a"]
    if row.size:
        lines.append(f"zeros: {n_zero / row.size:.0%} of patches")
    ax.text(0.97, 0.95, "\n".join(lines), transform=ax.transAxes, ha="right",
            va="top", fontsize=6.5, color=TICK_LABEL_COLOR, linespacing=1.3)

    ax.set_yticks([])
    ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{v:.2g}"))
    ax.tick_params(axis="x", labelsize=6.5)
    tidy_axes(ax, spines=("bottom",))


def _shorten(text, n):
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + "…"


def _fraction(x):
    """``1/6`` for 0.1666..., else the plain number."""
    from fractions import Fraction

    f = Fraction(x).limit_denominator(12)
    return f"{f.numerator}/{f.denominator}" if abs(float(f) - x) < 1e-9 else f"{x:.3g}"


def _log_ticks(xlim, *, minor=False):
    """Major (or minor) tick positions for a log axis spanning ``xlim``."""
    lo, hi = sorted(xlim)
    exps = np.arange(np.floor(np.log10(lo)) - 1, np.ceil(np.log10(hi)) + 2)
    every = [m * 10.0**e for e in exps for m in range(1, 10)]

    def in_view(ticks):
        return sum(lo <= t <= hi for t in ticks)

    major = [10.0**e for e in exps]
    if in_view(major) > 7:
        major = [10.0**e for e in exps if e % 2 == 0]
    else:
        for subs in ((1, 5), (1, 2, 5), range(1, 10)):
            if in_view(major) >= 2:
                break
            major = [m * 10.0**e for e in exps for m in subs]
    if not minor:
        return major
    taken = {round(np.log10(t), 6) for t in major}
    return [t for t in every if round(np.log10(t), 6) not in taken]
