"""Loader and CLI smoke tests on a small synthetic matrix."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vulntool import compute_vulnerability, load_matrix, bin_census, census_matrices
from vulntool.cli import main as cli_main
from vulntool.io import keep_rows, status_like_columns, suggest_row_filter
from vulntool.pipeline import _compute_from_matrices


def _make_csv(tmp_path):
    rng = np.random.default_rng(0)
    species = [f"Sp{i}" for i in range(6)]
    n_patch = 40
    rows = []
    for j in range(n_patch):
        vals = rng.gamma(shape=2.0, scale=5.0, size=len(species))
        vals[rng.random(len(species)) < 0.3] = 0.0  # some absences
        rows.append([f"P{j}"] + list(np.round(vals, 2)))
    df = pd.DataFrame(rows, columns=["patch"] + species)
    path = tmp_path / "synthetic.csv"
    df.to_csv(path, index=False)
    return path, species


def test_load_matrix_header(tmp_path):
    path, species = _make_csv(tmp_path)
    data, names, patches = load_matrix(path)
    assert names == species
    assert data.shape == (6, 40)
    assert len(patches) == 40


def test_pipeline_runs_on_synthetic(tmp_path):
    path, _ = _make_csv(tmp_path)
    result = compute_vulnerability(path, frac_keep=1.0, max_zero_frac=1.0)
    assert len(result.table) == 6
    assert result.M0 > 0
    assert set(["species", "p_alpha", "betabar", "delta", "W_alpha"]).issubset(result.table.columns)
    assert np.isfinite(result.table["W_alpha"]).all()


def _counts_and_sizes(tmp_path):
    """A pair of tables where size and headcount disagree about what is common.

    ``Sp5`` is a handful of very large trees: it dominates by size and barely
    registers by count, so a 95% cut keeps it in one table and drops it from the
    other. That disagreement is the whole reason ``selection_source`` exists.
    """
    rng = np.random.default_rng(3)
    species = [f"Sp{i}" for i in range(6)]
    n_patch = 40
    counts = rng.integers(20, 60, size=(n_patch, len(species))).astype(float)
    counts[:, 5] = rng.integers(1, 3, size=n_patch)      # rare by headcount
    # Mean tree size per species; Sp5's are giants, so it dominates by size.
    sizes = counts * np.array([1.0, 1.5, 2.0, 2.5, 3.0, 400.0])

    paths = []
    for name, values in (("counts.csv", counts), ("sizes.csv", sizes)):
        df = pd.DataFrame(values, columns=species)
        df.insert(0, "patch", [f"P{j}" for j in range(n_patch)])
        path = tmp_path / name
        df.to_csv(path, index=False)
        paths.append(path)
    return paths


def test_selection_source_changes_which_species_are_kept(tmp_path):
    counts, sizes = _counts_and_sizes(tmp_path)

    on_itself = compute_vulnerability(sizes)
    on_counts = compute_vulnerability(sizes, selection_source=counts)

    # Selected on size, the giant is the first thing kept; selected on counts it
    # falls in the discarded tail.
    assert "Sp5" in set(on_itself.table["species"])
    assert "Sp5" not in set(on_counts.table["species"])

    # Only the choice of species moved: the fit still runs on the size-weighted
    # table, so the abundances reported are size totals, not headcounts.
    sizes_df = pd.read_csv(sizes).set_index("patch")
    kept = on_counts.table.set_index("species")["abundance"]
    for name, total in kept.items():
        assert total == pytest.approx(sizes_df[name].sum())


def test_selection_source_with_mismatched_species_is_refused(tmp_path):
    counts, sizes = _counts_and_sizes(tmp_path)

    shuffled = pd.read_csv(counts)
    shuffled = shuffled[[shuffled.columns[0], *reversed(list(shuffled.columns[1:]))]]
    shuffled_path = tmp_path / "shuffled.csv"
    shuffled.to_csv(shuffled_path, index=False)

    with pytest.raises(ValueError, match="different order"):
        compute_vulnerability(sizes, selection_source=shuffled_path)

    renamed = pd.read_csv(counts).rename(columns={"Sp0": "Other"})
    renamed_path = tmp_path / "renamed.csv"
    renamed.to_csv(renamed_path, index=False)

    with pytest.raises(ValueError, match="different species"):
        compute_vulnerability(sizes, selection_source=renamed_path)


def test_unlabelled_inputs_skip_the_species_name_check(tmp_path):
    """Bare arrays carry placeholder names, so there is nothing to compare."""
    counts, sizes = _counts_and_sizes(tmp_path)
    a = load_matrix(sizes)[0]
    b = load_matrix(counts)[0]
    result = compute_vulnerability(a, selection_source=b)
    assert len(result.table) > 0


def test_cli_fit_writes_csv(tmp_path):
    path, _ = _make_csv(tmp_path)
    out = tmp_path / "out.csv"
    rc = cli_main(["fit", str(path), "-o", str(out),
                   "--frac-keep", "1.0", "--max-zero-frac", "1.0"])
    assert rc == 0
    assert out.exists()
    written = pd.read_csv(out)
    assert "W_alpha" in written.columns


def test_bin_census():
    stems = pd.DataFrame({
        "x": [1.0, 2.0, 12.0, 13.0, 1.0],
        "y": [1.0, 3.0, 1.0, 2.0, 12.0],
        "species": ["A", "A", "B", "A", "B"],
    })
    data, names, patches = bin_census(stems, L=10.0, extent=(0.0, 20.0, 0.0, 20.0))
    assert set(names) == {"A", "B"}
    assert data.sum() == 5  # five stems total
    assert data.shape[1] == 4  # full 2x2 grid (exact division), incl. empty cells
    assert len(patches) == 4


def test_bin_census_requires_extent():
    stems = pd.DataFrame({"x": [1.0], "y": [1.0], "species": ["A"]})
    with pytest.raises(ValueError):
        bin_census(stems, L=10.0, extent=None)


def test_bin_census_exact_tiling_no_ragged_edge():
    # L=31 on a 1000x500 plot -> nx=round(1000/31)=32, ny=round(500/31)=16,
    # dx=dy=31.25 exactly; every patch equal-area, none partial.
    rng = np.random.default_rng(0)
    n = 40000
    df = pd.DataFrame({
        "x": rng.uniform(0, 1000, n),
        "y": rng.uniform(0, 500, n),
        "species": "A",
    })
    data, names, patches = bin_census(df, L=31.0, extent=(0, 1000, 0, 500))
    assert data.shape[1] == 32 * 16  # full exact grid
    assert data.sum() == n           # no stems dropped


def test_census_matrices_keeps_species_with_no_valid_dbh():
    """A species whose every stem is missing DBH must not vanish from the
    crown matrix -- it needs a zero row so crown and stem-count matrices agree
    on the species set (previously an AssertionError in census_matrices)."""
    stems = pd.DataFrame({
        "x": [1.0, 2.0, 12.0, 13.0],
        "y": [1.0, 3.0, 1.0, 2.0],
        "species": ["A", "A", "B", "B"],
        "dbh": [10.0, 12.0, np.nan, np.nan],  # species B has no valid dbh
    })
    with pytest.warns(UserWarning, match="non-finite value"):
        crown, counts, names, binned = census_matrices(
            stems, L=10.0, extent=(0.0, 20.0, 0.0, 20.0), dbh="dbh",
        )
    assert names == ["A", "B"]
    assert crown.shape == counts.shape
    b = names.index("B")
    assert crown[b].sum() == 0      # no crown mass: dbh was missing throughout
    assert counts[b].sum() == 2     # but the two stems still count


def test_selected_species_with_zero_fit_total_is_refused():
    """If a species passes selection on stem counts but has zero total on the
    fit matrix (e.g. DBH missing for all its stems), fitting it would silently
    read as "absent everywhere" -- refuse instead."""
    rng = np.random.default_rng(0)
    n_patches = 20
    species = ["common", "sparse"]
    # "common" has real crown mass everywhere; "sparse" has stem counts big
    # enough to survive selection but zero crown mass (all DBH missing).
    counts = np.array([
        rng.integers(5, 10, n_patches),
        rng.integers(1, 3, n_patches),
    ], dtype=float)
    crown = np.array([
        counts[0] * 3.5,
        np.zeros(n_patches),
    ], dtype=float)
    with pytest.raises(ValueError, match="zero total in the fit matrix"):
        _compute_from_matrices(crown, counts, species, frac_keep=1.0, max_zero_frac=1.0)


def test_cli_fit_writes_species_fits_pdf(tmp_path):
    path, _ = _make_csv(tmp_path)
    out = tmp_path / "fits.pdf"
    rc = cli_main(["fit", str(path), "-o", str(tmp_path / "out.csv"),
                   "--frac-keep", "1.0", "--max-zero-frac", "1.0",
                   "--fits-pdf", str(out)])
    assert rc == 0
    assert out.read_bytes().startswith(b"%PDF")


# --------------------------------------------------------------------------
# ForestGEO-style census tables: dead stems, status filters, any separator
# --------------------------------------------------------------------------
def _forestgeo_table():
    """The demo census under ForestGEO column names (it already carries a
    ForestGEO status column), and its live stems alone."""
    from vulntool import load_demo_census

    table = load_demo_census().rename(columns={"x": "gx", "y": "gy", "species": "sp"})
    return keep_rows(table, "status", ["A"]), table


def test_the_demo_census_is_mostly_alive_and_dead_stems_have_no_dbh():
    alive, table = _forestgeo_table()
    assert set(table.status) == {"A", "D", "M"}
    assert len(alive) == 30_000
    assert len(alive) / len(table) > 0.85
    assert alive.dbh.notna().all()
    assert table.loc[table.status != "A", "dbh"].isna().all()


def test_keep_rows_compares_as_text():
    df = pd.DataFrame({"status": ["A", "D", "A", np.nan], "code": [1, 2, 1, 1]})
    assert len(keep_rows(df, "status", ["A"])) == 2
    assert len(keep_rows(df, "code", ["1"])) == 3
    assert len(keep_rows(df, "code", [1])) == 3
    assert len(keep_rows(df, "status", ["A", "nan"])) == 3


def test_suggest_row_filter_recognises_forestgeo_status():
    _, table = _forestgeo_table()
    assert suggest_row_filter(table) == ("status", ("A",))
    spelled = table.rename(columns={"status": "DFstatus"}).replace(
        {"DFstatus": {"A": "alive", "D": "dead"}})
    assert suggest_row_filter(spelled) == ("DFstatus", ("alive",))


def test_suggest_row_filter_stays_out_of_the_way():
    alive, _ = _forestgeo_table()
    assert suggest_row_filter(alive) is None                       # nothing to drop
    assert suggest_row_filter(alive.drop(columns="status")) is None
    codes = alive.drop(columns="status").assign(codes="M")         # site-specific
    assert suggest_row_filter(codes) is None


def test_status_like_columns_are_flagged_by_name():
    alive, table = _forestgeo_table()
    assert status_like_columns(table) == ["status"]
    assert status_like_columns(alive) == []          # every row already alive
    codes = table.rename(columns={"status": "Codes"})
    assert status_like_columns(codes) == ["Codes"]
    assert suggest_row_filter(codes) is None         # flagged, never selected
    assert status_like_columns(alive.drop(columns="status")) == []


def test_cli_census_reads_tab_separated_text_and_filters_dead_stems(tmp_path, capsys):
    from vulntool.demo import DEMO_PLOT

    alive, table = _forestgeo_table()
    txt = tmp_path / "census.txt"
    table.to_csv(txt, sep="\t", index=False)
    ref = tmp_path / "alive.csv"
    alive.to_csv(ref, index=False)

    plot = ["--extent", *map(str, DEMO_PLOT.extent)]
    for hole in DEMO_PLOT.holes:
        plot += ["--exclude", *map(str, hole)]
    common = [*plot, "--x", "gx", "--y", "gy", "--species", "sp", "--L", "50"]

    assert cli_main(["census", str(txt), *common, "-o", str(tmp_path / "all.csv")]) == 0
    assert "--filter status=A" in capsys.readouterr().err

    codes = tmp_path / "codes.txt"
    table.rename(columns={"status": "codes"}).to_csv(codes, sep="\t", index=False)
    assert cli_main(["census", str(codes), *common, "-o", str(tmp_path / "c.csv")]) == 0
    err = capsys.readouterr().err
    assert "'codes' may record whether each stem is alive" in err
    assert "--filter codes=<values meaning alive>" in err

    assert cli_main(["census", str(txt), *common, "--filter", "status=A",
                     "-o", str(tmp_path / "live.csv")]) == 0
    assert cli_main(["census", str(ref), *common, "-o", str(tmp_path / "ref.csv")]) == 0
    live = pd.read_csv(tmp_path / "live.csv")
    pd.testing.assert_frame_equal(live, pd.read_csv(tmp_path / "ref.csv"))
    unfiltered = pd.read_csv(tmp_path / "all.csv")
    assert unfiltered["abundance"].sum() > live["abundance"].sum()
