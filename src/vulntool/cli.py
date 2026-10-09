"""Command-line interface: ``vulnerability``.

    vulnerability fit CENSUS.csv -o results.csv [--plot out.png] [--fits-pdf fits.pdf]
    vulnerability census TREES.csv --extent 0 1000 0 500 -o results.csv
    vulnerability gui               # open the browser app (optional extra)
"""
from __future__ import annotations

import argparse
import sys

from .citation import CITATION_TEXT, REQUEST
from .gof import CVM_REFERENCE
from .io import DBH_UNITS
from .pipeline import compute_vulnerability, compute_vulnerability_census
from .selection import FRAC_KEEP, MAX_ZERO_FRAC


def _cmd_fit(args):
    result = compute_vulnerability(
        args.census,
        selection_source=args.selection,
        frac_keep=args.frac_keep,
        max_zero_frac=args.max_zero_frac,
        allow_nan=args.allow_nan,
        verbose=args.verbose,
    )
    print(f"Species kept = {len(result.table)} | patch capacity = {result.M0:.6g} "
          f"| abundance spread = {result.cv_xs:.4g}")

    if args.output:
        result.to_csv(args.output)
        print(f"Wrote per-species table -> {args.output}")
    else:
        print(result.table.to_string(index=False))

    if args.plot:
        from .plotting import plot_vulnerability
        import matplotlib
        matplotlib.use("Agg")
        ax = plot_vulnerability(result, annotate=args.annotate)
        ax.figure.savefig(args.plot, dpi=150)
        print(f"Wrote plot -> {args.plot}")
    _write_fits(result, args)
    return 0


def _cmd_census(args):
    import pandas as pd

    from .footprint import normalize_footprint
    from .io import keep_rows, status_like_columns, suggest_row_filter

    stems = pd.read_csv(args.stems, sep=args.sep, engine="python")
    if args.filter:
        col, _, allowed = args.filter.partition("=")
        if col not in stems.columns:
            raise SystemExit(
                f"--filter column {col!r} not found. Available columns: "
                f"{', '.join(map(str, stems.columns))}"
            )
        keep = [v for v in allowed.split(",") if v]
        stems = keep_rows(stems, col, keep)
        if stems.empty:
            raise SystemExit(f"--filter {args.filter!r} kept no stems.")
        print(f"Filter {col} in {keep}: kept {len(stems)} stems")
    else:
        # Census tables often list dead or missing stems too; counting them as
        # trees would be wrong without any error, so point it out (but do not
        # decide for the user).
        columns = status_like_columns(stems)
        if columns:
            hint = suggest_row_filter(stems)
            example = (f"--filter {hint[0]}={hint[1][0]}" if hint
                       else f"--filter {columns[0]}=<values meaning alive>")
            names = ", ".join(repr(str(c)) for c in columns)
            print(f"Note: column(s) {names} may record whether each stem is alive. "
                  "No --filter was given, so every row is counted as a live tree. "
                  f"To keep only live stems, use e.g. {example}.",
                  file=sys.stderr)

    # --footprint may be repeated; each occurrence is one rectangle of the union.
    # --exclude is the complementary spelling: rectangles cut back out of it.
    if args.footprint:
        rects = [tuple(r) for r in args.footprint]
        if len(rects) == 1:
            rects = rects[0]
    else:
        rects = tuple(args.extent)
    footprint = normalize_footprint(
        rects,
        bbox=tuple(args.extent) if args.extent else None,
        holes=[tuple(h) for h in args.exclude] if args.exclude else None,
    )

    result, scan = compute_vulnerability_census(
        stems,
        L=args.L,
        footprint=footprint,
        x=args.x, y=args.y, species=args.species, dbh=args.dbh,
        dbh_scale=args.dbh_scale, dbh_units=args.dbh_units,
        frac_keep=args.frac_keep, max_zero_frac=args.max_zero_frac,
        allow_nan=args.allow_nan,
        verbose=True,
    )

    if scan is not None:
        print()
        print(scan.display_table().to_string(index=False))
        print(f"(misfit is how far the fitted spread is from the real one for the "
              f"typical species; below {CVM_REFERENCE:.3g} counts as a good fit)")
    print(f"\nSpecies kept = {len(result.table)} | patch capacity = {result.M0_label} "
          f"| fit quality = {result.cvm:.4g} (good below {CVM_REFERENCE:.3g})")

    if args.output:
        result.to_csv(args.output)
        print(f"Wrote per-species table -> {args.output}")
    else:
        print(result.table.to_string(index=False))

    if args.gof_plot and scan is not None:
        from .plotting import plot_gof_vs_L
        import matplotlib
        matplotlib.use("Agg")
        ax = plot_gof_vs_L(scan)
        ax.figure.savefig(args.gof_plot, dpi=150)
        print(f"Wrote scale-selection plot -> {args.gof_plot}")

    if args.plot:
        from .plotting import plot_vulnerability
        import matplotlib
        matplotlib.use("Agg")
        ax = plot_vulnerability(result, annotate=args.annotate)
        ax.figure.savefig(args.plot, dpi=150)
        print(f"Wrote plot -> {args.plot}")
    _write_fits(result, args)
    return 0


def _write_fits(result, args):
    if not args.fits_pdf:
        return
    import matplotlib
    matplotlib.use("Agg")
    from .plotting import save_species_fits_pdf
    save_species_fits_pdf(result, args.fits_pdf)
    print(f"Wrote per-species fits -> {args.fits_pdf}")


def _cmd_gui(args):
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "The GUI needs the optional 'gui' extra (Streamlit). Add it with:\n"
            '    pip install "vulntool[gui] @ git+https://github.com/davideb321/vulntool"\n'
            "then run:  vulnerability gui\n"
            "(the double quotes matter: [gui] is a wildcard in zsh and PowerShell)",
            file=sys.stderr,
        )
        return 1
    import os
    from streamlit.web import cli as stcli
    gui_path = os.path.join(os.path.dirname(__file__), "_gui_app.py")
    sys.argv = ["streamlit", "run", gui_path]
    return stcli.main()


def build_parser():
    # The citation goes in the epilog rather than being printed after every run:
    # it is there for anyone who looks, and in the way of nobody.
    p = argparse.ArgumentParser(
        prog="vulnerability",
        description="Estimate how exposed each tree species is to being crowded "
                    "out locally, from a single forest census.",
        epilog=f"{REQUEST}\n\n  {CITATION_TEXT}\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fit", help="analyse a census already divided into patches")
    f.add_argument("census", help="census CSV: one row per patch, species names "
                                  "along the top, patch label in the first column")
    f.add_argument("-o", "--output", help="write per-species table to this CSV (else print)")
    f.add_argument("--selection", help="a second matrix (typically tree counts) used "
                                       "only to decide which species to keep; by "
                                       "default that is decided from the census "
                                       "matrix itself")
    f.add_argument("--frac-keep", type=float, default=FRAC_KEEP, dest="frac_keep",
                   help="keep the commonest species until they account for this "
                        f"share of all individuals (default {FRAC_KEEP})")
    f.add_argument("--max-zero-frac", type=float, default=MAX_ZERO_FRAC, dest="max_zero_frac",
                   help="drop species missing from more than this share of "
                        f"patches (default {MAX_ZERO_FRAC})")
    f.add_argument("--allow-nan", action="store_true", dest="allow_nan",
                   help="treat blank or non-numeric cells as 'none here' instead "
                        "of refusing to run")
    f.add_argument("--plot", help="also write the vulnerability-vs-abundance chart to this PNG")
    f.add_argument("--fits-pdf", dest="fits_pdf",
                   help="also write the per-species fits (histogram and "
                        "fitted density) to this PDF, ordered by "
                        "descending abundance")
    f.add_argument("--annotate", type=int, default=0,
                   help="label this many of the most vulnerable species on the chart")
    f.add_argument("-v", "--verbose", action="store_true")
    f.set_defaults(func=_cmd_fit)

    c = sub.add_parser(
        "census",
        help="start from a list of individual trees; work out the patch size too",
    )
    c.add_argument("stems", help="tree-list CSV: one row per tree (x, y, species[, dbh])")
    c.add_argument("--sep", default=None,
                   help="column separator of the tree list; detected from the file "
                        r"if not given (comma, tab, space, ...). E.g. '\t' or '\s+'")
    c.add_argument("--filter", metavar="COL=V1,V2",
                   help="keep only trees whose COL is one of the listed values, "
                        "e.g. status=A to drop dead ones")
    c.add_argument("--extent", type=float, nargs=4,
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                   help="the surveyed bounds, for a rectangular plot; "
                        "e.g. --extent 0 1000 0 500")
    c.add_argument("--footprint", type=float, nargs=4, action="append",
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                   help="one surveyed block of an irregularly shaped plot; repeat "
                        "for each. Together they make up the surveyed area, and "
                        "patches outside all of them are set aside rather than "
                        "recorded as ground where nothing grew.")
    c.add_argument("--exclude", type=float, nargs=4, action="append",
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                   help="an area to CUT OUT of the plot — a lake, a road, a "
                        "neighbouring property; repeat for each. Often the easier "
                        "way to describe an odd shape, and exactly equivalent to "
                        "listing the blocks that remain.")
    c.add_argument("--dbh-units", default=None, dest="dbh_units",
                   choices=sorted(DBH_UNITS),
                   help="units the DBH column is recorded in (default mm). Getting "
                        "this wrong does not change the vulnerability numbers; it "
                        "only changes the units the patch capacity is reported in.")
    c.add_argument("--dbh-scale", type=float, default=None, dest="dbh_scale",
                   help="for DBH units --dbh-units does not cover: the factor that "
                        "converts your diameters to millimetres (default 1.0)")
    c.add_argument("-o", "--output", help="write per-species table to this CSV (else print)")
    c.add_argument("--L", type=float, default=None,
                   help="use this patch size instead of working one out")
    c.add_argument("--x", default="x", help="x-coordinate column (default 'x')")
    c.add_argument("--y", default="y", help="y-coordinate column (default 'y')")
    c.add_argument("--species", default="species", help="species column (default 'species')")
    c.add_argument("--dbh", default=None,
                   help="trunk-diameter column. If given, a species' presence in a "
                        "patch is measured by the canopy area its trees cover rather "
                        "than by how many there are, so one big tree counts for more "
                        "than one sapling. Which species to keep is still decided on "
                        "counts.")
    c.add_argument("--frac-keep", type=float, default=FRAC_KEEP, dest="frac_keep",
                   help="keep the commonest species until they account for this "
                        f"share of all individuals (default {FRAC_KEEP})")
    c.add_argument("--max-zero-frac", type=float, default=MAX_ZERO_FRAC, dest="max_zero_frac",
                   help="drop species missing from more than this share of "
                        f"patches (default {MAX_ZERO_FRAC})")
    c.add_argument("--allow-nan", action="store_true", dest="allow_nan",
                   help="treat blank or non-numeric cells as 'none here' instead "
                        "of refusing to run")
    c.add_argument("--gof-plot", dest="gof_plot",
                   help="write the chart of fit quality against patch size to this PNG")
    c.add_argument("--plot", help="also write the vulnerability-vs-abundance chart to this PNG")
    c.add_argument("--fits-pdf", dest="fits_pdf",
                   help="also write the per-species fits (histogram and "
                        "fitted density) to this PDF, ordered by "
                        "descending abundance")
    c.add_argument("--annotate", type=int, default=0,
                   help="label this many of the most vulnerable species on the chart")
    c.set_defaults(func=_cmd_census)

    g = sub.add_parser("gui", help="open the point-and-click app in your browser "
                                   "(needs the [gui] extra)")
    g.set_defaults(func=_cmd_gui)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) == "census" and not args.extent and not args.footprint:
        if args.exclude:
            parser.error(
                "--exclude removes rectangles from the plot outline, so it needs "
                "--extent XMIN XMAX YMIN YMAX (or --footprint) to remove them from."
            )
        parser.error(
            "the census command needs the plot outline: --extent XMIN XMAX YMIN YMAX "
            "for a rectangular plot, or repeated --footprint rectangles (or "
            "--extent plus --exclude) for an irregular one. It cannot be inferred "
            "from the stem positions."
        )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
