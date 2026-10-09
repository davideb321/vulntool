"""The GUI's pure helpers, exercised without streamlit installed.

``vulntool.gui`` keeps ``import streamlit`` inside ``run()`` precisely so this
module can be imported and its parsing/plotting helpers tested in CI, where the
optional [gui] extra is not installed.
"""
from __future__ import annotations

import io

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from vulntool.gui import (
    CENSUS_DEFAULTS,
    DEMO_TABLE_L,
    FOOTPRINT_MODES,
    OUTSIDE_COLOR,
    SAMPLED_COLOR,
    ZERO_COLOR,
    DRAFT_COLOR,
    build_footprint,
    column_problems,
    data_extent,
    default_new_rect,
    demo_settings,
    demo_upload,
    file_signature,
    footprint_figure,
    footprint_key,
    outer_range_figure,
    guess_columns,
    parse_footprint_text,
    patch_colors,
    patch_totals,
    preview_census,
    rect_label,
    rect_problem,
    rects_to_text,
    scale_stats,
)

MICHIGAN = [
    (-100.0, 300.0, 0.0, 400.0),
    (-200.0, -100.0, 100.0, 400.0),
    (-300.0, -200.0, 100.0, 200.0),
    (300.0, 400.0, 0.0, 200.0),
    (400.0, 500.0, 0.0, 100.0),
]
#: The same plot spelled as what is *missing* from its bounding box.
MICHIGAN_HOLES = [
    (-300.0, -100.0, 0.0, 100.0),
    (-300.0, -200.0, 200.0, 400.0),
    (300.0, 400.0, 200.0, 400.0),
    (400.0, 500.0, 100.0, 400.0),
]


def test_importing_gui_does_not_need_streamlit():
    import vulntool.gui as gui
    assert callable(gui.run)


def test_parse_rectangles_commas_and_spaces():
    assert parse_footprint_text("0, 1000, 0, 500") == [(0.0, 1000.0, 0.0, 500.0)]
    assert parse_footprint_text("0 1000 0 500") == [(0.0, 1000.0, 0.0, 500.0)]


def test_parse_multiple_with_blanks_and_comments():
    text = """
    # Michigan
    -100, 300, 0, 400
    -200, -100, 100, 400

    -300, -200, 100, 200   # west arm
    """
    assert parse_footprint_text(text) == MICHIGAN[:3]


def test_parse_empty_is_none():
    assert parse_footprint_text("") is None
    assert parse_footprint_text("   \n # only a comment\n") is None


@pytest.mark.parametrize("bad,msg", [
    ("0, 1000, 0", "needs 4"),
    ("0, 1000, 0, 500, 7", "needs 4"),
    ("0, 1000, zero, 500", "non-numeric"),
])
def test_parse_errors_name_the_line(bad, msg):
    with pytest.raises(ValueError, match=msg):
        parse_footprint_text(bad)


def _cell_counts(ax):
    """(sampled, total) patch cells drawn — the outlines have no fill."""
    from matplotlib.colors import to_rgba

    cells = [p for p in ax.patches if p.get_fill()]
    sampled = [p for p in cells if p.get_facecolor() == to_rgba(SAMPLED_COLOR)]
    return len(sampled), len(cells)


def test_footprint_figure_draws_sampled_patches():
    ax = footprint_figure(MICHIGAN, 50.0)
    # The preview must show the same 92-of-128 the pipeline uses.
    assert _cell_counts(ax) == (92, 128)
    assert ax.get_title() == ""


def test_footprint_figure_rectangular_plot_is_fully_sampled():
    ax = footprint_figure((0.0, 1000.0, 0.0, 500.0), 50.0)
    assert _cell_counts(ax) == (200, 200)


# --------------------------------------------------------------------------
# The binned preview: patch totals, colours, and what they are allowed to mean.
# --------------------------------------------------------------------------
def _stems(seed=0, n=400):
    """A small clustered census in a 100 x 100 plot with a 20 x 20 clearing."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(seed)
    # Cluster around a few centres so patches differ in occupancy; uniform
    # stems would have no spatial structure, which is the one thing measured.
    centres = rng.uniform(5, 95, size=(8, 2))
    pts = centres[rng.integers(0, len(centres), n)] + rng.normal(0, 8, size=(n, 2))
    pts = np.clip(pts, 0.01, 99.99)
    return pd.DataFrame({
        "x": pts[:, 0],
        "y": pts[:, 1],
        "species": rng.choice(list("ABCD"), n),
        "dbh": rng.uniform(10, 500, n),
    })


SQUARE_BBOX = "0, 100, 0, 100"
SQUARE_EXTENT = (0.0, 100.0, 0.0, 100.0)
SQUARE_HOLE = "40, 60, 40, 60"


def test_patch_totals_scatters_onto_the_full_grid_with_nan_outside():
    from vulntool.io import bin_census

    fp = build_footprint(SQUARE_BBOX, holes_text=SQUARE_HOLE, mode="holes")
    binned = bin_census(_stems(), 10.0, footprint=fp)
    totals, mask = patch_totals(binned)

    assert totals.size == 100 and mask.sum() == 96
    assert np.isnan(totals[~mask]).all()          # dropped ground is not a zero
    assert not np.isnan(totals[mask]).any()
    assert totals[mask].sum() == binned.data.sum()


def test_empty_sampled_patches_are_zero_not_nan():
    """The distinction the whole preview exists to make."""
    pd = pytest.importorskip("pandas")
    stems = pd.DataFrame({"x": [5.0, 5.0], "y": [5.0, 5.0], "species": ["A", "A"]})
    fp = build_footprint(SQUARE_BBOX, mode="bbox")
    preview, _ = preview_census(stems, fp, 10.0)

    assert preview.n_empty == 99          # only the corner patch is occupied
    assert (preview.totals[preview.mask] == 0).sum() == 99
    assert not np.isnan(preview.totals).any()


def test_preview_is_always_stem_counts_never_the_crown_metric():
    """A DBH column must not change the picture — a big tree is not many trees."""
    stems = _stems()
    fp = build_footprint(SQUARE_BBOX, mode="bbox")
    preview, _ = preview_census(stems, fp, 20.0)
    assert preview.n_stems == len(stems)
    assert preview.totals[preview.mask].sum() == len(stems)
    # every total is a whole number of stems, not a sum of dbh**(4/3)
    assert np.all(preview.totals[preview.mask] == np.round(preview.totals[preview.mask]))


def test_preview_reports_stems_dropped_outside_the_outline():
    pd = pytest.importorskip("pandas")
    stems = pd.DataFrame({"x": [5.0, 5000.0], "y": [5.0, 5.0], "species": ["A", "B"]})
    fp = build_footprint(SQUARE_BBOX, mode="bbox")
    preview, warned = preview_census(stems, fp, 10.0)
    assert preview.n_stems == 1
    assert any("outside the extent" in w for w in warned)


def test_patch_colors_geometry_mode_is_the_old_two_colour_scheme():
    from matplotlib.colors import to_rgba

    mask = np.array([True, False, True])
    colors = patch_colors(mask)
    np.testing.assert_allclose(colors[0], to_rgba(SAMPLED_COLOR))
    np.testing.assert_allclose(colors[1], to_rgba(OUTSIDE_COLOR))


def test_patch_colors_separates_empty_from_low_and_from_dropped():
    from matplotlib.colors import to_rgba

    mask = np.array([True, True, True, False])
    values = np.array([0.0, 1.0, 50.0, np.nan])
    colors = patch_colors(mask, values)

    np.testing.assert_allclose(colors[0], to_rgba(ZERO_COLOR))
    np.testing.assert_allclose(colors[3], to_rgba(OUTSIDE_COLOR))
    # A count of 1 must not be confusable with "empty": it sits on the ramp,
    # far from the sentinel colour.
    assert np.abs(colors[1][:3] - np.asarray(to_rgba(ZERO_COLOR))[:3]).max() > 0.3
    assert not np.allclose(colors[1], colors[0])


def test_patch_colors_is_monotone_in_count():
    mask = np.ones(3, dtype=bool)
    colors = patch_colors(mask, np.array([1.0, 10.0, 100.0]))
    luminance = colors[:, :3].mean(axis=1)
    assert luminance[0] > luminance[1] > luminance[2]   # light -> dark


def test_patch_colors_handles_a_single_distinct_count():
    """One value must not land on the near-white bottom of the ramp."""
    colors = patch_colors(np.ones(2, dtype=bool), np.array([7.0, 7.0]))
    assert colors[:, :3].mean() < 0.5


def test_patch_colors_rejects_values_of_the_wrong_length():
    with pytest.raises(ValueError, match="full grid"):
        patch_colors(np.ones(10, dtype=bool), np.ones(4))


def test_footprint_figure_shaded_flags_empty_patches():
    fp = build_footprint("-300, 500, 0, 400", rects_text="\n".join(
        f"{a},{b},{c},{d}" for a, b, c, d in MICHIGAN), mode="rects")
    totals = np.full(128, np.nan)
    from vulntool.footprint import grid_shape, valid_mask
    nx, ny, _, _ = grid_shape(fp, 50.0)
    mask = valid_mask(fp, nx, ny)
    totals[mask] = 3.0
    totals[np.flatnonzero(mask)[:2]] = 0.0

    ax = footprint_figure(fp, 50.0, values=totals)
    assert len([p for p in ax.patches if p.get_fill()]) == 128

    # The empty state is hatched as well as coloured, in the map and the legend.
    hatched = [p for p in ax.patches if p.get_hatch()]
    assert len(hatched) == 2
    assert any(h.get_hatch() for h in ax.get_legend().get_patches())


def test_scale_stats_agrees_with_the_footprint_geometry():
    from vulntool.footprint import admissible_scales, grid_shape, valid_mask

    fp = build_footprint(SQUARE_BBOX, holes_text=SQUARE_HOLE, mode="holes")
    scales = admissible_scales(fp, min_L=10.0, min_patches=25)
    stats = scale_stats(_stems(), fp, scales)

    assert list(stats["patch size"]) == list(scales)
    for _, row in stats.iterrows():
        nx, ny, _, _ = grid_shape(fp, row["patch size"])
        assert row["patches surveyed"] == valid_mask(fp, nx, ny).sum()
        assert row["patches surveyed"] + row["patches set aside"] == nx * ny
        assert 0 <= row["% with none"] <= 100


# --------------------------------------------------------------------------
# Footprint assembly from the GUI's textareas
# --------------------------------------------------------------------------
def test_build_footprint_modes():
    assert build_footprint(SQUARE_BBOX, mode="bbox").is_rectangle
    holed = build_footprint(SQUARE_BBOX, holes_text=SQUARE_HOLE, mode="holes")
    assert holed.area() == pytest.approx(9600)
    rected = build_footprint(SQUARE_BBOX, rects_text="0, 50, 0, 100", mode="rects")
    assert rected.area() == pytest.approx(5000)


def test_build_footprint_holes_mode_needs_a_rectangle():
    with pytest.raises(ValueError, match="at least one"):
        build_footprint(SQUARE_BBOX, mode="holes")


def test_build_footprint_rejects_a_multi_line_bounding_box():
    with pytest.raises(ValueError, match="single line"):
        build_footprint("0, 100, 0, 100\n0, 50, 0, 50", mode="bbox")


def test_build_footprint_requires_a_bounding_box():
    with pytest.raises(ValueError, match="four numbers"):
        build_footprint("", mode="bbox")


def test_build_footprint_michigan_spellings_agree():
    from vulntool.footprint import grid_shape, valid_mask

    a = build_footprint("-300, 500, 0, 400", mode="rects", rects_text="\n".join(
        f"{r[0]},{r[1]},{r[2]},{r[3]}" for r in MICHIGAN))
    b = build_footprint("-300, 500, 0, 400", mode="holes", holes_text="\n".join(
        f"{r[0]},{r[1]},{r[2]},{r[3]}" for r in MICHIGAN_HOLES))
    for L in (10.0, 20.0, 25.0, 50.0, 100.0):
        nx, ny, _, _ = grid_shape(a, L)
        np.testing.assert_array_equal(valid_mask(a, nx, ny), valid_mask(b, nx, ny))


def test_footprint_key_is_hashable_and_spelling_specific():
    a = build_footprint("-300, 500, 0, 400", mode="rects", rects_text="\n".join(
        f"{r[0]},{r[1]},{r[2]},{r[3]}" for r in MICHIGAN))
    b = build_footprint("-300, 500, 0, 400", mode="holes", holes_text="\n".join(
        f"{r[0]},{r[1]},{r[2]},{r[3]}" for r in MICHIGAN_HOLES))
    assert hash(footprint_key(a)) and hash(footprint_key(b))
    # Same shape, different objects: cache keys track identity, not equivalence.
    assert footprint_key(a) != footprint_key(b)
    assert footprint_key(a) == footprint_key(a)


def test_file_signature_of_nothing_uploaded():
    assert file_signature(None) is None


# --------------------------------------------------------------------------
# Which column is which
# --------------------------------------------------------------------------
def test_guess_columns_finds_the_common_spellings():
    assert guess_columns(["x", "y", "species", "dbh"]) == {
        "xcol": "x", "ycol": "y", "spcol": "species", "dbhcol": "dbh"}
    forestgeo = "tag,treeid,stemtag,spcode,quadrat,gx,gy,dbh,codes,hom,date"
    assert guess_columns(forestgeo.split(",")) == {
        "xcol": "gx", "ycol": "gy", "spcol": "spcode", "dbhcol": "dbh"}


def test_guess_columns_ignores_case_and_keeps_the_header_spelling():
    assert guess_columns(["GX", "GY", "Species"])["xcol"] == "GX"
    assert guess_columns(["GX", "GY", "Species"])["spcol"] == "Species"


def test_guess_columns_falls_back_to_distinct_columns_and_no_dbh():
    assert guess_columns(["a", "b", "c", "d"]) == {
        "xcol": "a", "ycol": "b", "spcol": "c", "dbhcol": ""}
    # A recognised name is never handed out twice.
    g = guess_columns(["x", "foo", "bar"])
    assert len({g["xcol"], g["ycol"], g["spcol"]}) == 3


def test_column_problems():
    stems = _stems()
    assert column_problems(stems, ("x", "y", "species"), "dbh") == []
    assert column_problems(stems, ("x", "y", "species")) == []
    assert "more than one of x, y" in column_problems(stems, ("x", "x", "species"))[0]
    assert "x column *species*" in column_problems(stems, ("species", "y", "x"))[0]
    assert "both as DBH" in column_problems(stems, ("x", "y", "species"), "x")[0]
    assert "DBH column *species*" in column_problems(
        stems, ("x", "y", "dbh"), "species")[0]


# --------------------------------------------------------------------------
# The example census the demo button loads
# --------------------------------------------------------------------------
def test_demo_settings_spell_out_the_plot_the_demo_was_simulated_in():
    """The button must fill in the outline that matches the data.

    A stem list carries no outline, so loading one without the other would hand
    the user an example that is quietly wrong — the north-east quarter counted
    as surveyed-but-empty rather than never surveyed.
    """
    from vulntool import DEMO_PLOT

    s = demo_settings()
    bbox = ", ".join(f"{s[k]:g}" for k in ("xmin", "xmax", "ymin", "ymax"))
    fp = build_footprint(bbox, holes_text=s["holes_text"],
                         rects_text="", mode=FOOTPRINT_MODES[s["fp_mode"]])
    assert fp == DEMO_PLOT.footprint


def test_demo_settings_only_touch_widgets_that_exist():
    assert set(demo_settings()) <= set(CENSUS_DEFAULTS)


def test_demo_upload_behaves_like_an_uploaded_file():
    """It travels through the census path as a real upload, cache keys included."""
    import io

    pd = pytest.importorskip("pandas")

    up = demo_upload()
    assert file_signature(up) == (up.file_id, up.name, up.size)
    stems = pd.read_csv(io.BytesIO(up.getvalue()))
    assert list(stems.columns) == ["x", "y", "species", "dbh", "status"]
    assert len(stems) > 1000
    # getvalue() is repeatable; read_csv on a live buffer would not be.
    assert up.getvalue() == up.getvalue()


def test_demo_patch_table_is_the_matrix_input_format():
    from vulntool import load_demo_patch_table

    table = load_demo_patch_table()
    assert table.columns[0] == "patch"
    assert table["patch"].is_unique
    # The unsurveyed quarter is absent, not a block of zero rows.
    assert len(table) == 80
    assert (table.iloc[:, 1:].to_numpy() >= 0).all()


# --------------------------------------------------------------------------
# Actually executing the app needs the optional [gui] extra.
# --------------------------------------------------------------------------
def _app():
    pytest.importorskip("streamlit")
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    entry = Path(__file__).resolve().parent.parent / "src/vulntool/_gui_app.py"
    return AppTest.from_file(str(entry), default_timeout=90)


def test_app_runs():
    """Regression: `streamlit run` executes the target as a top-level script.

    Pointing it at a module with relative imports raises ImportError before a
    single widget renders — and the server still answers HTTP 200, so only
    executing the script catches it.
    """
    at = _app().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.title[0].value.startswith("🌳 vulntool v")


def test_the_app_opens_on_the_tree_list():
    """The census path is the primary mode; a patch table is that path part-done."""
    at = _app().run()
    assert at.radio[0].value == "List of individual trees"
    assert at.radio[0].options[0] == "List of individual trees"


# --------------------------------------------------------------------------
# The stepper, driven end to end:
#     1 Load -> 2 Outline -> 3 Calculate -> 4 Results
# --------------------------------------------------------------------------
def _square_census_csv(**rename):
    """A small clustered census in a 100 x 100 plot, as uploadable CSV bytes."""
    return _stems(seed=1, n=600).rename(columns=rename).to_csv(index=False).encode("utf-8")


def _labels(at):
    return [b.label for b in at.button]


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _click(at, label):
    _button(at, label).click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _back(at):
    return _click(at, "← Back")


def _step(at, key="step"):
    return at.session_state[key]


def _subheaders(at):
    return [s.value for s in at.subheader]


def _metrics(at):
    return {m.label: m.value for m in at.metric}


NEXT_1 = "Next — the plot outline"
NEXT_2 = "Looks right — choose the patch size"
NEXT_3 = "Compute vulnerability"


def _loaded_app(csv=None, name="stems.csv"):
    """Census mode, step 1, a stem list uploaded and DBH set to none."""
    at = _app().run()
    at.radio[0].set_value("List of individual trees").run()
    at.file_uploader[0].set_value((name, csv or _square_census_csv(), "text/csv"))
    at.run()
    at.selectbox(key="dbhcol").set_value("").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


EXTENT_KEYS = ("xmin", "xmax", "ymin", "ymax")


def _set_extent(at, *extent):
    for key, value in zip(EXTENT_KEYS, extent):
        at.number_input(key=key).set_value(float(value))
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _extent(at):
    return tuple(at.number_input(key=k).value for k in EXTENT_KEYS)


def _square_app():
    """Step 2, with the stem list loaded and the 0-100 square outline entered."""
    return _set_extent(_click(_loaded_app(), NEXT_1), *SQUARE_EXTENT)


def test_step_one_is_the_file_alone():
    """No outline yet, and nothing to move on to without a file."""
    at = _app().run()
    assert _step(at) == 1
    assert _subheaders(at) == ["Step 1 — load the tree list"]
    assert _button(at, NEXT_1).disabled
    assert not at.text_input   # the outline belongs to step 2
    assert "← Back" not in _labels(at)


def test_the_header_is_on_the_first_screen_only():
    at = _square_app()
    assert not at.title
    _back(at)
    assert at.title[0].value.startswith("🌳 vulntool v")

    at = _click(_matrix_mode(), "Load an example table")
    assert at.title
    _click(at, "Compute vulnerability")
    assert not at.title


def test_columns_are_offered_from_the_file_and_pre_guessed():
    at = _loaded_app(_square_census_csv(x="gx", y="gy", species="spcode"))
    assert at.selectbox(key="xcol").options == ["gx", "gy", "spcode", "dbh"]
    assert at.selectbox(key="xcol").value == "gx"
    assert at.selectbox(key="ycol").value == "gy"
    assert at.selectbox(key="spcol").value == "spcode"
    assert any("`gx`" in m.value for m in at.markdown)   # the headers are listed
    assert not _button(at, NEXT_1).disabled


def test_the_dbh_column_is_guessed_too():
    at = _app().run()
    at.file_uploader[0].set_value(("stems.csv", _square_census_csv(), "text/csv"))
    at.run()
    assert at.selectbox(key="dbhcol").value == "dbh"


def test_a_new_file_brings_fresh_guesses():
    at = _loaded_app(_square_census_csv(x="gx", y="gy", species="spcode"))
    at.file_uploader[0].set_value(("other.csv", _square_census_csv(), "text/csv"))
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.selectbox(key="xcol").value == "x"
    assert at.selectbox(key="spcol").value == "species"


def test_an_impossible_column_choice_blocks_next():
    at = _loaded_app()
    at.selectbox(key="ycol").set_value("x").run()
    assert any("more than one of x, y" in e.value for e in at.error)
    assert _button(at, NEXT_1).disabled

    at.selectbox(key="ycol").set_value("y")
    at.selectbox(key="xcol").set_value("species").run()
    at.selectbox(key="spcol").set_value("x").run()
    assert any("are not numbers" in e.value for e in at.error)
    assert _button(at, NEXT_1).disabled


def test_next_shows_the_outline_and_the_patches_at_once():
    """No separate binning button: the data is known, so the picture is live."""
    at = _square_app()
    assert _step(at) == 2
    assert _subheaders(at) == ["Step 2 — outline the plot and check the patches"]
    assert not at.file_uploader   # step 1 is off screen, not merely scrolled past
    metrics = _metrics(at)
    assert int(metrics["Patches surveyed"]) > 0
    assert metrics["Trees placed"] == "600"
    assert int(metrics["Patches with no trees"]) < int(metrics["Patches surveyed"])
    assert NEXT_2 in _labels(at)
    assert NEXT_3 not in _labels(at)
    assert any("lie within x" in c.value for c in at.caption)


def test_editing_the_outline_redraws_the_patches():
    at = _square_app()
    before = int(_metrics(at)["Patches surveyed"])
    at.radio(key="fp_mode").set_value("A rectangle with pieces cut out").run()
    at.text_area(key="holes_text").set_value(SQUARE_HOLE).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert int(_metrics(at)["Patches surveyed"]) < before


def test_the_extent_boxes_step_by_fifty():
    at = _square_app()
    assert all(at.number_input(key=k).step == 50 for k in EXTENT_KEYS)
    at.number_input(key="xmax").increment().run()
    assert at.session_state["xmax"] == 150


def test_step_two_says_the_outline_sets_the_patch_sizes():
    at = _square_app()
    assert any("computed from the outline" in i.value for i in at.info)


def test_an_outline_too_small_for_50_m_patches_is_warned_about():
    at = _square_app()   # 100 x 100: at least 25 patches caps L at 20
    assert max(_scale_options(at)) in (10, 20)
    assert any("patch size is small" in w.value for w in at.warning)

    _set_extent(at, 0, 500, 0, 500)
    assert not any("patch size is small" in w.value for w in at.warning)


def _scale_options(at):
    """The preview's patch sizes as numbers (they are shown as "50 m")."""
    return [float(o.removesuffix(" m")) for o in at.selectbox(key="preview_L").options]


def test_from_cannot_be_stepped_past_to():
    at = _square_app()
    _set_extent(at, 100, 100, 0, 100)    # 'from' pushed up to 'to'
    assert _extent(at) == (50, 100, 0, 100)
    _set_extent(at, 50, 100, 0, -20)     # 'to' pushed below 'from'
    assert _extent(at) == (50, 100, 0, 50)
    assert not at.error
    assert NEXT_2 in _labels(at)


def test_figures_can_be_saved():
    at = _square_app()
    buttons = at.get("download_button")
    assert [b.proto.label for b in buttons] == ["Save PNG", "Save PDF"]
    assert all(b.proto.ignore_rerun for b in buttons)


def test_uploaded_file_survives_repeated_reruns():
    """Regression: read_csv consumes an UploadedFile's buffer.

    Handing the UploadedFile straight to read_csv works exactly once; on the
    next rerun it raises EmptyDataError, which is why the parse goes through
    getvalue() instead.
    """
    at = _square_app()
    for _ in range(3):
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
    assert _step(at) == 2
    assert NEXT_2 in _labels(at)


def test_back_keeps_the_file_the_columns_and_the_outline():
    """The uploader forgets its file the moment it is not rendered; the app
    must not. Back from step 2 lands on a filled-in step 1, and Next goes
    straight through again to the outline as it was."""
    at = _back(_square_app())
    assert _step(at) == 1
    assert at.file_uploader[0].value is None           # the widget forgot...
    assert any("stems.csv" in c.value for c in at.caption)   # ...the app did not
    assert at.selectbox(key="xcol").value == "x"
    assert at.selectbox(key="dbhcol").value == ""

    _click(at, NEXT_1)
    assert _step(at) == 2
    assert _extent(at) == SQUARE_EXTENT
    assert _metrics(at)["Trees placed"] == "600"


def test_forgetting_the_kept_file_empties_step_one():
    at = _back(_square_app())
    _click(at, "Forget this file")
    assert not any("stems.csv" in c.value for c in at.caption)
    assert any("Upload your tree list" in i.value for i in at.info)
    assert _button(at, NEXT_1).disabled


def test_step_three_works_from_what_step_two_showed_last():
    at = _set_extent(_square_app(), 0, 200, 0, 200)
    scales = _scale_options(at)
    at.selectbox(key="preview_L").set_value(scales[0]).run()
    _click(at, NEXT_2)
    setup = at.session_state["setup"]
    assert setup.footprint.width == 200
    assert setup.preview_L == float(scales[0])

    _back(at)
    _set_extent(at, *SQUARE_EXTENT)
    _click(at, NEXT_2)
    assert at.session_state["setup"].footprint.width == 100


def _dbh_step_three():
    at = _loaded_app()
    at.selectbox(key="dbhcol").set_value("dbh").run()
    _set_extent(_click(at, NEXT_1), *SQUARE_EXTENT)
    return _click(at, NEXT_2)


def test_with_extra_scale():
    from vulntool.gui import with_extra_scale

    fp = build_footprint("0, 1000, 0, 500")
    ladder = (10.0, 20.0, 50.0)
    assert with_extra_scale(fp, ladder, "") == (ladder, None, None)
    assert with_extra_scale(fp, ladder, " 31.25 m") == ((10.0, 20.0, 31.25, 50.0), 31.25, None)
    assert with_extra_scale(fp, ladder, "50") == (ladder, 50.0, None)
    scales, extra, error = with_extra_scale(fp, ladder, "33")
    assert scales == ladder and extra is None and "not offered" in error


def test_a_typed_patch_size_joins_the_ladder_everywhere():
    at = _square_app()
    assert 5.0 not in _scale_options(at)
    at.text_input(key="extra_L").input("5").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert 5.0 in _scale_options(at)
    assert at.selectbox(key="preview_L").value == 5.0     # shown straight away
    assert _metrics(at)["Patches surveyed"] == "400"

    _click(at, NEXT_2)
    assert 5.0 in at.session_state["setup"].scales
    assert "5" in at.radio(key="fit_L").options


def test_a_typed_patch_size_that_does_not_tile_is_refused():
    at = _square_app()
    ladder = _scale_options(at)
    at.text_input(key="extra_L").input("7").run()
    assert _scale_options(at) == ladder
    assert any("does not divide the plot evenly" in e.value for e in at.error)
    assert NEXT_2 in _labels(at)                           # not a blocker


def test_step_three_offers_auto_and_every_admissible_patch_size():
    at = _click(_square_app(), NEXT_2)
    radio = at.radio(key="fit_L")
    assert radio.value == "auto"
    assert radio.options[1:] == [f"{v:g}" for v in at.session_state["setup"].scales]


def test_a_dbh_column_can_be_switched_on_and_off_in_step_three():
    at = _dbh_step_three()
    assert at.session_state["setup"].dbhcol == "dbh"
    assert at.toggle(key="use_dbh").value is True
    assert at.selectbox(key="dbh_units")   # units only matter with DBH on
    at.radio(key="fit_L").set_value(at.session_state["setup"].scales[0]).run()
    _click(at, NEXT_3)
    assert not _metrics(at)["Patch capacity"].endswith("stems")

    _back(at)
    at.toggle(key="use_dbh").set_value(False).run()
    assert not [s for s in at.selectbox if s.key == "dbh_units"]
    _click(at, NEXT_3)
    assert _metrics(at)["Patch capacity"].endswith("stems")


def test_without_a_dbh_column_there_is_nothing_to_switch():
    at = _click(_square_app(), NEXT_2)
    assert not at.toggle
    assert any("choose a DBH column in step 1" in c.value for c in at.caption)


def test_every_step_has_a_back_button_and_only_step_one_does_not():
    at = _square_app()
    assert "← Back" in _labels(at)
    _click(at, NEXT_2)
    assert _step(at) == 3
    assert _subheaders(at) == ["Step 3 — calculate"]
    assert "← Back" in _labels(at)
    _back(at)
    assert _step(at) == 2
    _back(at)
    assert _step(at) == 1
    assert "← Back" not in _labels(at)


def _pick_smallest_scale(at):
    """Step 3: a fixed patch size, which skips the scale scan."""
    at.radio(key="fit_L").set_value(at.session_state["setup"].scales[0]).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def test_compute_runs_end_to_end_and_reports_units():
    at = _pick_smallest_scale(_click(_square_app(), NEXT_2))
    _click(at, NEXT_3)

    assert _step(at) == 4
    assert _subheaders(at)[0] == "Step 4 — results"
    metrics = _metrics(at)
    assert int(metrics["Species kept"]) > 0
    assert metrics["Patch capacity"].endswith("stems")


def test_results_survive_a_rerun_that_touches_an_unrelated_widget():
    at = _pick_smallest_scale(_click(_square_app(), NEXT_2))
    _click(at, NEXT_3)

    at.checkbox(key="show_labels").set_value(False).run()   # plot-only, not a fit input
    assert not at.exception, [str(e.value) for e in at.exception]
    assert "Species kept" in {m.label for m in at.metric}


def test_the_scale_choice_survives_the_trip_to_results_and_back():
    """fit_L lives on a widget step 4 does not render; it must persist."""
    at = _pick_smallest_scale(_click(_square_app(), NEXT_2))
    _click(at, NEXT_3)
    assert not any("Choosing the patch size" in m.value for m in at.markdown)
    _back(at)
    assert at.radio(key="fit_L").value == at.session_state["setup"].scales[0]


def _census_mode(at, *, mode="Several blocks joined together", rects=MICHIGAN):
    """Enter the Michigan outline on step 2."""
    at.radio(key="fp_mode").set_value(mode)
    _set_extent(at, -300, 500, 0, 400)
    key = "rects_text" if mode == "Several blocks joined together" else "holes_text"
    at.text_area(key=key).set_value("\n".join(f"{a},{b},{c},{d}" for a, b, c, d in rects))
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _area_caption(at):
    return next(c.value for c in at.caption if c.value.startswith("Surveyed area"))


def test_an_irregular_outline_is_measured_in_the_app():
    caption = _area_caption(_census_mode(_square_app()))
    assert "230,000" in caption and "320,000" in caption


def test_holes_and_rects_spellings_agree_in_the_app():
    rects = _census_mode(_square_app())
    holes = _census_mode(_square_app(), mode="A rectangle with pieces cut out",
                         rects=MICHIGAN_HOLES)
    assert _area_caption(rects) == _area_caption(holes)
    assert _metrics(rects) == _metrics(holes)


# --------------------------------------------------------------------------
# The demo buttons, driven end to end.
# --------------------------------------------------------------------------
def _demo_census_app():
    at = _app().run()
    at.radio[0].set_value("List of individual trees").run()
    return _click(at, "Load an example census")


def test_the_demo_goes_through_step_one_like_any_file():
    """The example shows how the tool is used, column choice included."""
    at = _demo_census_app()
    assert _step(at) == 1
    assert any("example census" in c.value for c in at.caption)
    assert {k: at.selectbox(key=k).value for k in ("xcol", "ycol", "spcol", "dbhcol")} \
        == {"xcol": "x", "ycol": "y", "spcol": "species", "dbhcol": "dbh"}
    assert not _button(at, NEXT_1).disabled


def test_the_demo_lists_every_automatic_choice_for_checking():
    at = _demo_census_app()
    (warning,) = [w.value for w in at.warning]
    for item in ("x position = `x`", "y position = `y`", "species = `species`",
                 "DBH = `dbh`", "rows kept where `status` is `A`"):
        assert item in warning


def test_the_demo_shows_the_live_stem_filter_at_work():
    """The example is a ForestGEO-style table: its dead and missing stems are
    left out by the pre-set filter, leaving the 30 000 live ones."""
    at = _demo_census_app()
    assert at.selectbox(key="filter_col").value == "status"
    assert at.multiselect(key="filter_vals").value == ["A"]
    assert at.multiselect(key="filter_vals").options == ["A", "D", "M"]
    assert any("Keeps 30,000 of 32,967 rows" in c.value for c in at.caption)
    assert any("30,000 are alive" in c.value for c in at.caption)


def test_the_demo_preview_shows_all_three_patch_states():
    """The example is only worth shipping if it exercises what the preview is
    for, so assert the three states are all present: patches dropped as
    unsurveyed (80 kept of a 96-cell grid), patches surveyed but empty (the
    clearing), and the occupied rest.
    """
    at = _click(_demo_census_app(), NEXT_1)
    assert _step(at) == 2
    metrics = _metrics(at)
    assert metrics["Patches surveyed"] == "80"
    assert 0 < int(metrics["Patches with no trees"]) < 80
    assert metrics["Trees placed"] == "30,000"
    assert NEXT_2 in _labels(at)


def test_demo_button_fills_in_the_outline_for_step_two():
    at = _click(_demo_census_app(), NEXT_1)
    s = demo_settings()
    assert _extent(at) == tuple(s[k] for k in EXTENT_KEYS)
    assert at.text_area(key="holes_text").value == s["holes_text"]
    assert at.radio(key="fp_mode").value == s["fp_mode"]


def test_uploading_a_file_replaces_the_demo():
    """The two must never be mixed: a real upload wins and the example clears."""
    at = _demo_census_app()
    at.file_uploader[0].set_value(
        ("stems.csv", _square_census_csv(x="gx", y="gy"), "text/csv"))
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert not any("example census" in c.value for c in at.caption)
    assert at.selectbox(key="xcol").value == "gx"


def test_the_census_form_survives_a_trip_through_the_other_input_mode():
    """Streamlit drops the state of widgets a rerun does not render.

    Left alone, that empties the outline while the loaded census stays — the
    example then paired with a default rectangle it was never surveyed on, which
    is exactly the silent mismatch the preview exists to catch.
    """
    at = _click(_demo_census_app(), NEXT_1)
    at.radio[0].set_value("Table of patches").run()
    at.radio[0].set_value("List of individual trees").run()

    assert not at.exception, [str(e.value) for e in at.exception]
    assert _step(at) == 2
    s = demo_settings()
    assert _extent(at) == tuple(s[k] for k in EXTENT_KEYS)
    assert at.text_area(key="holes_text").value == s["holes_text"]
    _back(at)
    assert at.selectbox(key="dbhcol").value == "dbh"


def test_demo_button_empties_the_uploader_rather_than_losing_to_it():
    """A file_uploader's value cannot be assigned, only re-keyed — so this can
    regress silently into the button appearing to do nothing but rewrite the
    outline."""
    at = _click(_loaded_app(), "Load an example census")
    assert at.file_uploader[0].value is None
    assert any("example census" in c.value for c in at.caption)
    assert not any("stems.csv" in c.value for c in at.caption)
    assert at.selectbox(key="xcol").value == "x"


def test_the_citation_footer_sits_in_the_sidebar_on_every_screen():
    from vulntool.citation import LINK

    def has_footer(at):
        return (any(LINK in c.value for c in at.sidebar.caption)
                and not any(LINK in c.value for c in at.main.caption))

    at = _app().run()
    assert has_footer(at)
    at = _square_app()
    assert has_footer(at)
    _click(at, NEXT_2)
    assert has_footer(at)
    assert has_footer(_matrix_mode())


def _matrix_mode():
    """App in patch-table mode (no longer the mode it opens on)."""
    at = _app().run()
    at.radio[0].set_value("Table of patches").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def test_demo_table_button_loads_an_example_in_matrix_mode():
    at = _click(_matrix_mode(), "Load an example table")
    assert any("Example loaded" in s.value for s in at.success)
    _click(at, "Compute vulnerability")
    assert _step(at, "table_step") == 2
    metrics = {m.label: m.value for m in at.metric}
    assert int(metrics["Species kept"]) > 0

    at.checkbox(key="show_labels").set_value(False).run()
    assert "Species kept" in {m.label for m in at.metric}
    _back(at)
    assert _step(at, "table_step") == 1
    assert any("Example loaded" in s.value for s in at.success)


def test_matrix_mode_keeps_an_uploaded_table_across_back():
    from vulntool import load_demo_patch_table

    table = load_demo_patch_table().to_csv(index=False).encode("utf-8")
    at = _matrix_mode()
    at.file_uploader[0].set_value(("counts.csv", table, "text/csv"))
    at.run()
    _click(at, "Compute vulnerability")
    _back(at)
    assert at.file_uploader[0].value is None
    assert any("counts.csv" in c.value for c in at.caption)
    _click(at, "Compute vulnerability")
    assert int({m.label: m.value for m in at.metric}["Species kept"]) > 0


def test_matrix_mode_offers_a_second_table_for_choosing_the_species():
    """A size-weighted table selected on itself keeps the wrong species, and
    nothing in the numbers reveals which kind of table it is — so the override
    has to be reachable from the app, not only from the library."""
    from vulntool import load_demo_patch_table

    at = _matrix_mode()
    assert len(at.file_uploader) == 2

    table = load_demo_patch_table().to_csv(index=False).encode("utf-8")
    at.file_uploader[0].set_value(("counts.csv", table, "text/csv"))
    at.file_uploader[1].set_value(("counts.csv", table, "text/csv"))
    at.run()
    _click(at, "Compute vulnerability")

    metrics = {m.label: m.value for m in at.metric}
    assert int(metrics["Species kept"]) > 0

    # Selecting counts on counts is a no-op: same answer as the single-table run.
    alone = _matrix_mode()
    alone.file_uploader[0].set_value(("counts.csv", table, "text/csv"))
    alone.run()
    _click(alone, "Compute vulnerability")
    assert {m.label: m.value for m in alone.metric} == metrics


def test_mismatched_second_table_is_refused_in_the_app():
    from vulntool import load_demo_patch_table

    table = load_demo_patch_table()
    shuffled = table[[table.columns[0], *reversed(list(table.columns[1:]))]]

    at = _matrix_mode()
    at.file_uploader[0].set_value(
        ("counts.csv", table.to_csv(index=False).encode("utf-8"), "text/csv"))
    at.file_uploader[1].set_value(
        ("shuffled.csv", shuffled.to_csv(index=False).encode("utf-8"), "text/csv"))
    at.run()
    _click(at, "Compute vulnerability")

    assert any("same species" in e.value for e in at.error)
    assert "Species kept" not in {m.label for m in at.metric}


# --------------------------------------------------------------------------
# Step 2 starts from the trees' range; blocks mode frames an empty outline
# --------------------------------------------------------------------------
def test_data_extent_rounds_the_tree_range_to_the_nearest_50():
    import pandas as pd

    stems = pd.DataFrame({"gx": [0.03, 599.77, np.nan], "gy": [-3.5, 2.0, 399.99]})
    assert data_extent(stems, "gx", "gy") == (0.0, 600.0, 0.0, 400.0)
    # Nearest, not outward: a stray tree 4 m past the edge does not move it.
    stray = pd.DataFrame({"gx": [-304.0, 498.2], "gy": [0.4, 377.0]})
    assert data_extent(stray, "gx", "gy") == (-300.0, 500.0, 0.0, 400.0)
    # Too narrow to round apart: widened outward to one step.
    narrow = pd.DataFrame({"gx": [10.0, 20.0], "gy": [0.0, 400.0]})
    assert data_extent(narrow, "gx", "gy")[:2] == (0.0, 50.0)
    assert data_extent(stems.assign(gx=np.nan), "gx", "gy") is None


def test_step_two_starts_from_the_trees_range():
    import io

    import pandas as pd

    csv = _square_census_csv()
    stems = pd.read_csv(io.BytesIO(csv))
    at = _click(_loaded_app(csv), NEXT_1)
    assert _extent(at) == data_extent(stems, "x", "y")
    assert _extent(at) != tuple(CENSUS_DEFAULTS[k] for k in EXTENT_KEYS)


def test_back_and_next_keep_an_edited_extent():
    """Only a new file, or new x/y columns, start the box over."""
    at = _square_app()
    _set_extent(at, 0, 150, 0, 100)
    _click(_back(at), NEXT_1)
    assert _extent(at) == (0.0, 150.0, 0.0, 100.0)


def test_a_file_after_the_demo_starts_from_its_own_range():
    at = _demo_census_app()
    at.file_uploader[0].set_value(("stems.csv", _square_census_csv(), "text/csv"))
    at.run()
    at.selectbox(key="dbhcol").set_value("").run()
    _click(at, NEXT_1)
    assert _extent(at)[1] <= 100.0


def test_blocks_mode_draws_the_outer_range_before_any_block():
    at = _square_app()
    at.radio(key="fp_mode").set_value("Several blocks joined together").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert not at.error
    assert any("Add at least one surveyed block" in i.value for i in at.info)
    assert NEXT_2 not in _labels(at)
    assert not at.metric

    at.text_area(key="rects_text").set_value("0, 100, 0, 50").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert NEXT_2 in _labels(at)
    assert _metrics(at)["Patches surveyed"]


def test_outer_range_figure_is_the_box_alone():
    ax = outer_range_figure((0.0, 300.0, 0.0, 200.0))
    assert len(ax.patches) == 1
    assert "no blocks yet" in ax.get_title()


# ------------------------------------------------- step 2: the area list
def test_rects_to_text_round_trips_and_labels_read_plainly():
    rects = [(-100.0, 300.0, 0.0, 400.0), (0.5, 50.0, 0.0, 50.0)]
    assert parse_footprint_text(rects_to_text(rects)) == rects
    assert rects_to_text([]) == ""
    assert rect_label(rects[0]) == "x -100 to 300 · y 0 to 400"


def test_a_new_area_starts_as_a_corner_square_inside_the_box():
    assert default_new_rect((-300, 500, 0, 400)) == (-300.0, -250.0, 0.0, 50.0)
    assert default_new_rect((0, 30, 0, 400)) == (0.0, 30.0, 0.0, 50.0)


@pytest.mark.parametrize("rect, others, mode, expected", [
    ((0, 50, 0, 50), (), "holes", None),
    ((50, 0, 0, 50), (), "holes", "larger than its 'from'"),
    ((0, 50, 50, 50), (), "rects", "larger than its 'from'"),
    ((-50, 50, 0, 50), (), "holes", "inside the outer range"),
    ((0, 50, 0, 150), (), "rects", "inside the outer range"),
    ((10, 20, 10, 20), [(0, 50, 0, 50)], "holes", "already cut out"),
    ((0, 100, 50, 100), [(0, 100, 0, 50)], "holes", "no surveyed ground"),
    ((0, 100, 0, 100), (), "holes", "no surveyed ground"),
    ((10, 20, 10, 20), [(0, 50, 0, 50)], "rects", "already covered"),
    ((25, 75, 0, 50), [(0, 50, 0, 50)], "rects", None),   # overlaps are fine
    ((25, 75, 0, 50), [(0, 50, 0, 50)], "holes", None),
])
def test_rect_problem(rect, others, mode, expected):
    problem = rect_problem(rect, (0, 100, 0, 100), others, mode)
    if expected is None:
        assert problem is None
    else:
        assert expected in problem


def test_the_draft_is_drawn_over_the_patches_and_named_in_the_legend():
    from matplotlib.colors import to_rgba

    fp = build_footprint(SQUARE_BBOX)
    plain = footprint_figure(fp, 20.0, values=np.ones(25))
    drafted = footprint_figure(fp, 20.0, values=np.ones(25),
                               draft=(0.0, 50.0, 0.0, 50.0))
    assert len(drafted.patches) == len(plain.patches) + 2
    assert drafted.patches[-1].get_edgecolor() == to_rgba(DRAFT_COLOR)
    labels = [t.get_text() for t in drafted.get_legend().get_texts()]
    assert "Area being entered" in labels

    # A draft straying outside the box is kept in view, to be seen to stray.
    far = footprint_figure(fp, 20.0, draft=(80.0, 160.0, 0.0, 50.0))
    assert far.get_xlim()[1] > 160.0

    ax = outer_range_figure((0.0, 300.0, 0.0, 200.0), draft=(0.0, 50.0, 0.0, 50.0))
    assert len(ax.patches) == 3


def _holes_mode(at):
    at.radio(key="fp_mode").set_value("A rectangle with pieces cut out").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _press(at, key):
    at.button(key=key).click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _set_draft(at, *rect):
    for k, v in zip(("xmin", "xmax", "ymin", "ymax"), rect):
        at.number_input(key=f"draft_{k}").set_value(float(v))
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _rows(at):
    return [m.value for m in at.markdown if m.value[:1].isdigit() and " · y " in m.value]


def test_holes_mode_starts_empty_and_surveys_the_whole_box():
    at = _holes_mode(_square_app())
    assert not at.error
    assert not _rows(at)
    assert any("Nothing cut out yet" in c.value for c in at.caption)
    assert _metrics(at)["Patches surveyed"]
    assert not _button(at, NEXT_2).disabled


def test_a_new_area_cannot_be_stepped_inside_out():
    at = _press(_holes_mode(_square_app()), "add_holes")
    _set_draft(at, 50, 50, 0, 50)
    assert tuple(at.number_input(key=f"draft_{k}").value
                 for k in ("xmin", "xmax", "ymin", "ymax")) == (0.0, 50.0, 0.0, 50.0)


def test_add_an_area_highlights_it_then_ok_puts_it_in_the_list():
    at = _press(_holes_mode(_square_app()), "add_holes")
    # The editor opens on a corner square, and the step waits for it.
    assert tuple(at.number_input(key=f"draft_{k}").value
                 for k in ("xmin", "xmax", "ymin", "ymax")) == (0.0, 50.0, 0.0, 50.0)
    assert at.number_input(key="draft_xmax").step == 50.0
    assert _button(at, NEXT_2).disabled
    assert at.text_area(key="holes_text").disabled
    before = _metrics(at)["Patches surveyed"]

    _set_draft(at, 0, 50, 0, 100)
    # Moving the draft re-draws, but does not re-bin: nothing is cut out yet.
    assert _metrics(at)["Patches surveyed"] == before
    _press(at, "ok_holes")
    assert at.session_state["holes_text"] == "0, 50, 0, 100"
    assert _rows(at) == ["1. x 0 to 50 · y 0 to 100"]
    assert not _button(at, NEXT_2).disabled
    assert any("Surveyed area 5,000 of 10,000" in c.value
                              for c in at.caption)


def test_edit_changes_the_item_and_cancel_leaves_it():
    at = _holes_mode(_square_app())
    at.text_area(key="holes_text").set_value("0, 50, 0, 50\n50, 100, 50, 100").run()
    assert len(_rows(at)) == 2

    _press(at, "edit_holes_1")
    assert at.number_input(key="draft_xmin").value == 50.0
    assert at.button(key="edit_holes_0").disabled   # one editor at a time
    assert "editing" in _rows(at)[1]
    _set_draft(at, 50, 100, 0, 100)
    _press(at, "ok_holes")
    assert at.session_state["holes_text"] == "0, 50, 0, 50\n50, 100, 0, 100"

    _press(at, "edit_holes_0")
    _set_draft(at, 0, 50, 0, 100)
    _press(at, "cancel_holes")
    assert at.session_state["holes_text"] == "0, 50, 0, 50\n50, 100, 0, 100"
    assert not _button(at, NEXT_2).disabled


def test_remove_takes_the_item_out():
    at = _holes_mode(_square_app())
    at.text_area(key="holes_text").set_value("0, 50, 0, 50\n50, 100, 50, 100").run()
    _press(at, "remove_holes_0")
    assert at.session_state["holes_text"] == "50, 100, 50, 100"
    assert _rows(at) == ["1. x 50 to 100 · y 50 to 100"]


def test_ok_refuses_an_impossible_area_and_keeps_the_editor_open():
    at = _press(_holes_mode(_square_app()), "add_holes")
    _set_draft(at, 0, 50, 0, 150)
    _press(at, "ok_holes")
    assert any("inside the outer range" in e.value for e in at.error)
    assert at.session_state["holes_text"] == ""
    assert at.button(key="ok_holes")
    _set_draft(at, 0, 50, 0, 50)
    _press(at, "ok_holes")
    assert not at.error
    assert at.session_state["holes_text"] == "0, 50, 0, 50"


def test_unreadable_text_is_reported_and_blocks_adding():
    at = _holes_mode(_square_app())
    at.text_area(key="holes_text").set_value("0, 50, 0").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("Edit as text" in e.value for e in at.error)
    assert at.button(key="add_holes").disabled
    assert NEXT_2 not in _labels(at)


def test_the_example_lists_its_own_cut_outs():
    at = _click(_demo_census_app(), NEXT_1)
    holes = parse_footprint_text(demo_settings()["holes_text"])
    assert _rows(at) == [f"{i}. {rect_label(h)}" for i, h in enumerate(holes, 1)]


def test_blocks_are_added_on_the_outer_range_frame():
    at = _square_app()
    at.radio(key="fp_mode").set_value("Several blocks joined together").run()
    _press(at, "add_rects")
    assert not at.error
    assert any("Add at least one surveyed block" in i.value for i in at.info)
    _set_draft(at, 0, 100, 0, 50)
    _press(at, "ok_rects")
    assert at.session_state["rects_text"] == "0, 100, 0, 50"
    assert _metrics(at)["Patches surveyed"]
    assert not _button(at, NEXT_2).disabled


def test_switching_shape_closes_an_open_editor():
    at = _press(_holes_mode(_square_app()), "add_holes")
    at.radio(key="fp_mode").set_value("Several blocks joined together").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.button(key="add_rects")
    assert at.session_state["holes_text"] == ""


def _processing(at):
    return [m.value for m in at.main.markdown if m.value.startswith("Processing ")]


def test_the_file_being_processed_is_named_past_step_one():
    at = _loaded_app()
    assert not _processing(at)                 # the uploader names it here
    _click(at, NEXT_1)
    assert _processing(at) == ["Processing `stems.csv`"]
    _click(at, NEXT_2)
    assert _processing(at) == ["Processing `stems.csv`"]

    at = _click(_loaded_app(), "Load an example census")
    _click(at, NEXT_1)
    assert _processing(at) == ["Processing the example census (`demo_census.csv`)"]

    at = _click(_matrix_mode(), "Load an example table")
    _click(at, "Compute vulnerability")
    assert _processing(at) == ["Processing the example table"]


def test_counts_colorbar_names_its_ends_in_plain_numbers():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    values = np.array([17.0, 40.0, 0.0, 146.0])
    footprint_figure((0.0, 40.0, 0.0, 40.0), 20.0, ax=ax, values=values,
                     colorbar=True)
    cax = fig.axes[-1]
    texts = {t.get_text() for t in cax.texts}
    assert texts == {"max", "146", "min", "17"}
    labels = [t.get_text() for t in cax.get_yticklabels()]
    assert labels
    assert all(t.replace(",", "").isdigit() for t in labels)
    plt.close(fig)


# --------------------------------------------------------------------------
# Per-species fits on the results page
# --------------------------------------------------------------------------
def _demo_results():
    at = _click(_matrix_mode(), "Load an example table")
    return _click(at, "Compute vulnerability")


def test_species_fits_are_off_until_asked_for():
    at = _demo_results()
    assert at.toggle(key="show_fits").value is False
    assert "Per-species fits" not in _subheaders(at)
    assert not [r for r in at.radio if r.key == "fits_order"]

    at.toggle(key="show_fits").set_value(True).run()
    assert "Per-species fits" in _subheaders(at)


def _fits_caption(at):
    return next(c.value for c in at.caption if "Each panel" in c.value)


def test_species_fits_mention_the_si_without_a_link():
    at = _demo_results()
    at.toggle(key="show_fits").set_value(True).run()
    caption = _fits_caption(at)
    assert "Supplementary Information" in caption
    assert "](" not in caption


def test_species_fits_page_through_every_species():
    from vulntool.gui import FITS_PER_PAGE

    at = _demo_results()
    n = int(_metrics(at)["Species kept"])
    assert n > FITS_PER_PAGE  # the demo exercises paging

    at.toggle(key="show_fits").set_value(True).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    pages = at.selectbox(key=f"fits_page_{n}")
    assert len(pages.options) == -(-n // FITS_PER_PAGE)
    assert pages.options[-1].endswith(f"of {n}")
    assert f"Save all {n} species (PDF)" in [b.label for b in at.get("download_button")]

    pages.set_value(len(pages.options) - 1).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    at.radio(key="fits_order").set_value("Descending cvm").run()
    assert not at.exception, [str(e.value) for e in at.exception]


# --------------------------------------------------------------------------
# ForestGEO-style tables in step 1: .txt uploads and the live-stem filter
# --------------------------------------------------------------------------
def _forestgeo_txt(n_dead=300):
    """The square census, tab-separated, in ForestGEO column names, with a
    status column and a dead-only species that only the filter leaves out."""
    pd = pytest.importorskip("pandas")
    alive = _stems(seed=1, n=600).rename(columns={"x": "gx", "y": "gy", "species": "sp"})
    alive["status"] = "A"
    dead = alive.iloc[:n_dead].assign(sp="Z", status="D", dbh=np.nan)
    table = pd.concat([alive, dead], ignore_index=True)
    table.loc[0, "status"] = np.nan        # BCI has a few blank status cells too
    return table.to_csv(sep="\t", index=False).encode("utf-8")


def _forestgeo_app():
    return _loaded_app(_forestgeo_txt(), name="census.txt")


def test_a_forestgeo_text_file_comes_in_filtered_to_live_stems():
    at = _forestgeo_app()
    assert at.selectbox(key="xcol").value == "gx"
    assert at.selectbox(key="spcol").value == "sp"
    assert at.selectbox(key="filter_col").value == "status"
    assert at.multiselect(key="filter_vals").value == ["A"]
    assert at.multiselect(key="filter_vals").options == ["A", "D", "nan"]
    assert any("Keeps 599 of 900 rows" in c.value for c in at.caption)
    (warning,) = [w.value for w in at.warning]
    assert warning.startswith("Selected automatically")
    for item in ("x position = `gx`", "species = `sp`",
                 "rows kept where `status` is `A`"):
        assert item in warning
    assert "DBH" not in warning      # _loaded_app already changed it to none

    _click(at, NEXT_1)
    assert at.session_state["loaded"].row_filter == ("status", ("A",))


def test_dropping_the_filter_is_allowed_but_flagged():
    at = _forestgeo_app()
    at.selectbox(key="filter_col").set_value("").run()
    assert any("`status` may record whether each stem is alive" in w.value
               for w in at.warning)
    assert not _button(at, NEXT_1).disabled
    _click(at, NEXT_1)
    assert at.session_state["loaded"].row_filter is None


def test_each_choice_leaves_the_automatic_list_once_changed():
    at = _forestgeo_app()
    warning = next(w.value for w in at.warning if "automatically" in w.value)
    assert "x position" in warning
    for key, value in (("xcol", "gy"), ("ycol", "gx"), ("spcol", "status")):
        at.selectbox(key=key).set_value(value)
    at.multiselect(key="filter_vals").set_value(["A", "nan"]).run()
    assert not [w for w in at.warning if "automatically" in w.value]


def test_a_codes_column_is_flagged_but_never_selected():
    """Condition codes vary from site to site: point the column out, but leave
    the choice of values to the user."""
    pd = pytest.importorskip("pandas")
    table = pd.read_csv(io.BytesIO(_forestgeo_txt()), sep="\t")
    table = table.rename(columns={"status": "codes"})
    at = _loaded_app(table.to_csv(index=False).encode(), name="codes.csv")
    assert at.selectbox(key="filter_col").value == ""
    assert any("`codes` may record whether each stem is alive" in w.value
               for w in at.warning)
    assert not _button(at, NEXT_1).disabled


def test_a_filter_that_keeps_nothing_blocks_next():
    at = _forestgeo_app()
    at.multiselect(key="filter_vals").set_value([]).run()
    assert any("No rows match" in e.value for e in at.error)
    assert _button(at, NEXT_1).disabled


def test_a_column_with_too_many_values_is_no_status_column():
    at = _forestgeo_app()
    at.selectbox(key="filter_col").set_value("gx").run()
    assert any("more than a status column usually has" in e.value for e in at.error)
    assert _button(at, NEXT_1).disabled


def test_the_filter_reaches_the_results():
    """The dead-only species is fitted without the filter, and gone with it."""
    def species_kept(filtered):
        at = _forestgeo_app()
        if not filtered:
            at.selectbox(key="filter_col").set_value("").run()
        at = _set_extent(_click(at, NEXT_1), *SQUARE_EXTENT)
        at = _click(_click(at, NEXT_2), NEXT_3)
        return int(_metrics(at)["Species kept"])

    assert species_kept(filtered=False) == species_kept(filtered=True) + 1


FORWARD = "Next →"


def test_forward_button_is_greyed_out_until_a_step_has_been_reached():
    at = _app().run()
    assert _button(at, FORWARD).disabled
    at = _square_app()                       # step 2, never left forwards
    assert _button(at, FORWARD).disabled
    _click(at, NEXT_2)
    assert _button(at, FORWARD).disabled     # step 3, nothing computed yet


def test_forward_returns_to_the_outline_while_step_one_is_unchanged():
    at = _back(_square_app())
    assert not _button(at, FORWARD).disabled
    _click(at, FORWARD)
    assert _step(at) == 2
    assert _extent(at) == SQUARE_EXTENT

    _back(at)
    at.selectbox(key="xcol").set_value("y").run()
    assert _button(at, FORWARD).disabled


def test_forward_returns_to_step_three_while_the_outline_is_unchanged():
    at = _click(_square_app(), NEXT_2)
    _back(at)
    assert not _button(at, FORWARD).disabled
    _set_extent(at, 0, 200, 0, 200)
    assert _button(at, FORWARD).disabled


def test_forward_returns_to_cached_results_until_a_fit_input_changes():
    at = _pick_smallest_scale(_click(_square_app(), NEXT_2))
    _click(at, NEXT_3)
    assert FORWARD not in _labels(at)        # nothing after the last step
    _back(at)
    assert not _button(at, FORWARD).disabled
    _click(at, FORWARD)
    assert _step(at) == 4
    assert "Species kept" in {m.label for m in at.metric}

    _back(at)
    at.radio(key="fit_L").set_value("auto").run()
    assert _button(at, FORWARD).disabled


def test_forward_in_table_mode_follows_the_table_and_settings():
    at = _matrix_mode()
    assert _button(at, FORWARD).disabled
    at = _demo_results()
    _back(at)
    assert not _button(at, FORWARD).disabled
    _click(at, FORWARD)
    assert _step(at, "table_step") == 2


def test_the_patch_size_curve_is_off_until_asked_for():
    at = _click(_square_app(), NEXT_2)       # automatic patch size: a scan
    _click(at, NEXT_3)
    assert any("Choosing the patch size" in m.value for m in at.markdown)
    assert at.toggle(key="show_gof_plot").value is False
    n_saves = len(at.get("download_button"))

    at.toggle(key="show_gof_plot").set_value(True).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert len(at.get("download_button")) == n_saves + 2   # Save PNG, Save PDF
