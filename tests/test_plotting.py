"""The vulnerability scatter's p_alpha axis stays readable at any span."""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from vulntool.plotting import plot_vulnerability  # noqa: E402


def _result(lo, hi, n=12):
    p = np.geomspace(lo, hi, n)
    table = pd.DataFrame({"species": [f"s{i}" for i in range(n)],
                          "p_alpha": p, "W_alpha": np.linspace(-1, 1, n)})
    return SimpleNamespace(table=table)


@pytest.mark.parametrize("lo, hi", [(2e-3, 6e-3), (1.2e-3, 9e-3), (3e-4, 8e-2), (1e-8, 1e-1)])
def test_p_alpha_axis_has_labels_and_enough_ticks(lo, hi):
    import matplotlib.pyplot as plt

    ax = plot_vulnerability(_result(lo, hi))
    ax.figure.canvas.draw()
    x0, x1 = ax.get_xlim()
    labelled = [t for t in ax.xaxis.get_major_ticks()
                if t.label1.get_visible() and t.label1.get_text()
                and x0 <= t.get_loc() <= x1]
    minor = [x for x in ax.xaxis.get_minorticklocs() if x0 <= x <= x1]
    assert 2 <= len(labelled) <= 8
    assert len(labelled) + len(minor) >= 5
    plt.close(ax.figure)


# --------------------------------------------------------------------------
# Per-species fit panels
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def demo_result():
    from vulntool import compute_vulnerability, load_demo_patch_table

    return compute_vulnerability(load_demo_patch_table())


def _visible_panels(fig):
    return [ax for ax in fig.axes if ax.get_visible()]


def test_result_keeps_the_values_that_were_fitted(demo_result):
    """One row per kept species, in table order; zero exactly where absent."""
    from vulntool import load_demo_patch_table
    from vulntool.theory import get_xs_from_data

    table = load_demo_patch_table().set_index("patch")
    kept = table[demo_result.table["species"]].to_numpy(float).T
    xs = demo_result.xs
    assert xs.shape == (len(demo_result.table), len(table))
    np.testing.assert_array_equal(xs == 0, kept == 0)
    np.testing.assert_allclose(xs, get_xs_from_data(kept, demo_result.M0)[0])


def test_species_fit_panels_follow_the_requested_rows(demo_result):
    import matplotlib.pyplot as plt

    from vulntool.plotting import plot_species_fits

    rows = [3, 0, 7]
    fig = plot_species_fits(demo_result, rows, ncols=2)
    panels = _visible_panels(fig)
    assert len(panels) == 3
    names = demo_result.table["species"]
    assert [ax.get_title(loc="left") for ax in panels] == [
        f"{i}  {names[i]}" for i in rows]
    plt.close(fig)


def test_species_fit_panel_states_parameters_cvm_and_zeros(demo_result):
    import matplotlib.pyplot as plt

    from vulntool.plotting import plot_species_fits

    row = demo_result.table.iloc[0]
    fig = plot_species_fits(demo_result, [0])
    (text,) = _visible_panels(fig)[0].texts
    share = float(np.mean(demo_result.xs[0] == 0))
    assert text.get_text().split("\n") == [
        rf"$\delta$ = {row['delta']:.3g}",
        rf"$\bar\beta$ = {row['betabar']:.3g}",
        f"cvm = {row['cvm']:.2g}",
        f"zeros: {share:.0%} of patches",
    ]
    plt.close(fig)


def test_cvm_is_not_highlighted_above_the_reference(demo_result):
    """1/6 is an approximate reference level, not a pass/fail threshold."""
    import copy

    import matplotlib.pyplot as plt

    from vulntool.plotting import plot_species_fits

    texts = []
    for value in (0.01, 0.5):
        r = copy.copy(demo_result)
        r.table = demo_result.table.copy()
        r.table.loc[0, "cvm"] = value
        fig = plot_species_fits(r, [0])
        (t,) = _visible_panels(fig)[0].texts
        texts.append((t.get_color(), t.get_fontweight()))
        plt.close(fig)
    assert texts[0] == texts[1]


def test_fit_order_defaults_to_the_table_and_can_put_worst_first(demo_result):
    import copy

    from vulntool.plotting import species_fit_order

    assert species_fit_order(demo_result) == list(demo_result.table.index)

    r = copy.copy(demo_result)
    r.table = demo_result.table.copy()
    r.table.loc[2, "cvm"] = np.nan
    worst = species_fit_order(r, worst_first=True)
    assert worst[-1] == 2
    scored = r.table.loc[worst[:-1], "cvm"].to_numpy()
    assert (np.diff(scored) <= 0).all()


def test_species_fits_need_the_fitted_values():
    from vulntool.plotting import plot_species_fits

    bare = SimpleNamespace(table=pd.DataFrame({"species": ["a"]}), xs=None)
    with pytest.raises(ValueError, match="result.xs"):
        plot_species_fits(bare)


def test_all_species_pdf_has_one_page_per_block(demo_result, tmp_path):
    from vulntool.plotting import save_species_fits_pdf

    path = save_species_fits_pdf(demo_result, tmp_path / "fits.pdf", per_page=10)
    import re

    pages = len(re.findall(rb"/Type\s*/Page(?!s)", path.read_bytes()))
    n = len(demo_result.table)
    assert pages == -(-n // 10)
