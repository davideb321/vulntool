# vulntool — user guide



## 1. Install options

Full install (with GUI based on Streamlit):

```bash
pip install "vulntool[gui] @ git+https://github.com/davideb321/vulntool"
```

Other install options:

```bash
# no GUI — just the library and the command line (numpy, scipy, pandas only)
pip install "vulntool @ git+https://github.com/davideb321/vulntool"

# + matplotlib for the diagnostic plots, without the GUI
pip install "vulntool[plot] @ git+https://github.com/davideb321/vulntool"
```

**No git?** `git+https://` needs git installed, which Windows
does not provide by default. GitHub also serves a plain archive, which pip
handles with no git involved:

```bash
pip install "vulntool[gui] @ https://github.com/davideb321/vulntool/archive/refs/heads/main.tar.gz"
```

**For reproducible work, pin a version.** The commands in the README track
whatever is on `main` the day you run them. Appending a released tag pins the
code instead, so an analysis can be repeated years later against the same
version:

```bash
pip install "vulntool[gui] @ git+https://github.com/davideb321/vulntool@v0.3.2"
```

#### From a clone

Only needed to read the source, modify it, or run the test suite:

```bash
git clone https://github.com/davideb321/vulntool && cd vulntool
pip install -e ".[gui,dev]"
pytest
```

`vulntool` is pure Python and nothing is compiled at install time, so all of the
above should work identically on Linux, macOS and Windows. Python 3.10 or newer.

#### Updating

If you installed with `pip install ... @ git+https://...` (any variant above),
add `--upgrade` to the same command to pull the latest `main`:

```bash
pip install --upgrade "vulntool[gui] @ git+https://github.com/davideb321/vulntool"
```

If you pinned a version (`@v0.3.2`), change the tag to upgrade to a specific
release instead of tracking `main`.

If you installed from a clone with `-e`, there's nothing to reinstall — just
pull:

```bash
git pull
```

Either way, the GUI's title bar shows the installed version, so you can
confirm the upgrade worked.

## 2. The demo census

Every install includes a small example census, so you can test the whole
pipeline with no data of your own. It is a synthetic 20 ha plot, 30 000 stems
across 40 species, simulated with a **Thomas process** (cluster centres
scattered over the plot, stems scattered around each centre, with a cluster
width that varies by species) — a clustered, patchy layout.

The file is laid out like a ForestGEO census table (see section 3): a `status`
column marks the 30 000 live stems `A`, and about 3 000 more rows record stems
that are dead (`D`) or missing (`M`), with no DBH. As with a real census, only
the live stems are analysed: the GUI selects them by default, and the examples
below show how to do the same elsewhere.

In the GUI it is one click — *Load an example* — which fills in the plot outline
along with the data.

From the command line, use the committed copy
[`examples/demo_census.csv`](examples/demo_census.csv):

```bash
vulnerability census examples/demo_census.csv \
    --extent 0 600 0 400 --exclude 400 600 200 400 --filter status=A \
    --dbh dbh --dbh-units mm -o results.csv --gof-plot scale.png
```

To import the census data in a script, use:

```python
from vulntool import load_demo_census, compute_vulnerability_census, DEMO_PLOT
from vulntool.io import keep_rows

stems = keep_rows(load_demo_census(), "status", ["A"])   # 30 000 live stems, 40 species, 20 ha
result, scan = compute_vulnerability_census(
    stems, footprint=DEMO_PLOT.footprint, dbh="dbh", dbh_units="mm",
)
print(scan.table)                  # the scale scan, with its recommendation
print(result.table)                # per-species W_alpha
```

The demo plot is a 600 x 400 m rectangle with its north-east quarter never
surveyed, plus a small treeless clearing inside the surveyed area — so the
dropped, empty-but-sampled and occupied patches all appear in the GUI preview.
`load_demo_patch_table()` returns the same census already divided into
patches, if what you want to see is the table input format (see Input
formats below).

## 3. Input formats

The tool loads a plain .csv or text file. Two cases are handled:

a) **If you have individual tree measurements**

The tool expects a file in which each row represents a single tree,
with at least three columns, two for the tree's spatial position (in metres) and one
for the species label or ID. Additionally, a fourth column can indicate each
tree's trunk diameter (DBH). Column labels can be specified at runtime
and additional columns can be present. Column separators (comma, tab, space,
semicolon) are inferred by the tool, so a `.txt` or `.tsv` file works as well as
a `.csv`.

Example:

```
x,y,species,dbh
12.4,301.9,Aglaia,84.0
55.1,17.2,Shorea,412.5
```

See `examples/demo_census.csv` for a full example. This is the case the tool is
built around: it divides the plot into patches for you, and works out how big
they should be. Refer to section 5 of the usage guide below.

**ForestGEO census tables.** Stem tables in the ForestGEO (CTFS) format can be
loaded as they are distributed, typically as tab- or space-separated `.txt`. The
columns `gx`, `gy`, `sp` (or `spcode`) and `dbh` are recognised automatically, and
DBH is taken to be in millimetres, the ForestGEO convention, unless you say
otherwise. One point needs attention: these tables have a row for every stem
ever tagged, including stems that have since **died** or could not be found,
and such a row must not be counted as a tree. Unfiltered, the 2005 BCI census
holds 145,467 dead stems next to 208,387 live ones, and keeping them would
change which species are analysed (113 instead of the 101 of the paper at
50 m patches). Each tool handles this as follows:

- The GUI recognises a ForestGEO `status` column (`A` alive, `D` dead,
  `M` missing; or `DFstatus` = `alive`) and keeps only live stems by default
  (step 1, **Which rows are live trees**), with a warning asking you to check
  the choice. If you remove that filter, a warning stays on screen.
- On the command line, pass `--filter status=A`. Without it, the tool prints a
  note when it sees a column that may record stem status (`status`, `codes`,
  `condition`, ...), but counts every row.
- In the library, filter the table first, e.g. with
  `vulntool.io.keep_rows(stems, "status", ["A"])`.

Some sites record a stem's condition in a different way; the Michigan Big Woods
census, for example, uses a `codes` column, where the paper kept the codes
`M`, `AL`, `B` and `R`. Such conventions differ from site to site and are not
guessed: choose the column and the values meaning alive yourself (GUI step 1,
or `--filter codes=M,AL,B,R`). The site's metadata documents them.

b) **If your data are already coarse-grained into spatial patches**

Then the tool expects one row per patch:
species names along the top, a patch label in the first column, and how much of
each species that patch holds in the cells. That can be a
count of trees, or something size-weighted such as total canopy area — anything
consistent across the table. The column separator is auto-detected (comma,
tab, semicolon, ...).

Example:

```
patch,Aglaia,Shorea,Dipterocarpus,Ficus
P00,0,0,19.1,10.4
P01,32.7,0,3.1,22.8
```

Patch labels are arbitrary; they are carried through but never interpreted.
See `examples/demo_patch_table.csv`.

This is case a) with the dividing-up already done, so the tool takes the patches
as given and skips the choice of patch size. Refer to section 6 of the usage
guide below.


## 4. Output description

The output of the tool is a table that has one row per species and these columns:

| Column | Description |
|---|---|
| `abundance` | the species' total across the whole plot |
| `p_alpha` | how much of an average patch it takes up, as a fraction — the plain "how common is it" |
| `delta` | how evenly it is spread. **Small means clumpy**: plenty in a few patches, none in most |
| `betabar` | the companion of `delta`; between them they describe the whole pattern of variation from patch to patch |
| `W_alpha` | **the vulnerability** — how exposed the species is. More negative is safer; near or above zero is the danger zone |
| `cvm` | how well the model matched this particular species (lower is better) |

Alongside the table, the tool computes the patch capacity, fitted from the data,
and a CvM proxy, indicating how well the model fitted overall.

The fit can also be inspected for each species. Each panel shows the histogram
of the vacancy-adjusted abundance over the patches where the species is present, together
with the fitted Gamma density, and reports the fitted parameters (`delta`,
`betabar`), the Cramér–von Mises statistic `cvm` and the fraction of patches where
the species is absent. These patches are excluded from the fit, so they are
reported as a number rather than drawn as a bar. For a correctly specified model
the expected value of `cvm` is approximately 1/6, which serves as an approximate
reference level. Panels are ordered by descending abundance, as in the paper's
supplement, or optionally by descending `cvm`.

See also the in-app description provided by the GUI.

## 5. Usage guide — starting from a tree census

This is the case the tool is built around: one row per tree, with its position,
its species, and optionally its trunk diameter. From that it divides the plot
into patches, decides how big those patches should be, and computes the
vulnerability of each species.

Two things have to be settled before any of that: how large a patch should be,
and which ground was actually surveyed. The tool works the first out from your
data, and needs to be told the second. Both are easiest to handle in the
**browser GUI**, where the patches are drawn for you and a mistyped outline is
visible immediately; the library and the command line do the same work without
the picture.

### Choosing the patch size

Small patches keep the spatial detail, but leave so few trees in each that the
model has little to go on and describes them badly. Large patches are described
well but average spatial details away, which is the thing you were trying to see.
So the tool tries a range of sizes, scores how well the model describes the data
at each, and takes **the smallest patches that are still described well**. The
score has a natural pass mark (`1/6`), and the table of sizes and scores is
always reported alongside the choice, so the decision can be checked rather than
taken on trust.

You can also fix the size yourself and skip the search, with `--L` on the command
line, `L=` in the library, or the patch-size selector in the GUI.

### The plot outline

**The outline of the plot is required.** It cannot be guessed from the trees,
because the outermost ones always stop short of the true edge — and getting it
wrong changes the answers, since how full a patch is depends on how big it is.

The plot is divided by exact division (`nx = round(W/L)` columns of width `W/nx`),
so patches always come out equal in size with no ragged strip along the edge,
whatever size you ask for.

Most real plots are not neat rectangles, and there are two equivalent ways to
describe one that is not. You can list the surveyed blocks, which together make
up the plot and can form an L, a staircase or anything else; or you can give the
enclosing rectangle and cut pieces out of it — a lake, a road, a neighbour's
land. The two spellings give identical results, and can be combined. Two blocks
that do not touch cannot be described by cutting pieces out, which is why both
forms exist.

A patch counts as part of the plot when its center falls inside the outline.
Patches outside are **set aside**, not recorded as empty, and that distinction
matters more than it sounds: treat an irregular plot as a plain rectangle and
nothing complains, but every cell of unsurveyed ground becomes ground where no
tree happened to grow. Every species then looks more absent than it is, and every
number that follows is affected.

For an irregular plot the grid also has to line up with the shape's own edges, so
only some patch sizes will do. The tool works out which from the geometry rather
than being told.

### Diameters, if you have them

Given a diameter column, a species' presence in a patch is measured by the canopy
area its trees cover rather than by how many there are, so one mature tree counts
for more than one sapling. **Which species to keep is still decided on tree
counts**, in all three interfaces, whether or not diameters were supplied — that
is the paper's policy, and on this path it is automatic.

Say which units your diameters are in if they are not millimetres (`mm`, `cm`,
`m`, `in`; or a scale factor of your own). Getting the units wrong will not change
the vulnerability numbers — the factor cancels — but it does change the units the
patch capacity is reported in, which is why that figure always carries its units
rather than appearing as a bare number.

### Browser GUI

The browser app covers both kinds of input. It runs entirely on your machine;
nothing is uploaded anywhere.

```bash
vulnerability gui        # after installing with the [gui] extra, see section 1
```

Starting from a list of trees, it goes in four steps, one on screen at a time
(the line at the top shows where you are; *← Back* returns to the previous step
with everything you entered still in place, including the uploaded file;
*Next →* returns to the following step when nothing on the current one has
changed since it was last left, so its results are reused rather than
recomputed; from step 2 on, the line below it names the file being processed):

1. **Load the tree list.** Upload the file (or load the example); the app lists
   its columns and shows the first rows. Pick which column holds the x and y
   positions, the species and, optionally, the trunk diameter, from dropdowns of
   the file's own column names — common names (`x`/`gx`, `y`/`gy`,
   `species`/`sp`/`spcode`, `dbh`) are pre-selected, so a typical census needs no
   picking at all. Under **Which rows are live trees** you can keep only the
   rows whose value in one column is among those you choose, to leave out dead
   or missing stems. Only an unambiguous marker is set automatically (a
   `status` column holding `A`, see section 3); other columns that may record a
   stem's status, such as `codes` or `condition`, are pointed out in a warning
   while no filter is set, but never selected. Every automatic choice, columns
   included, is listed in a warning until you change it. A choice that cannot work (the same column
   twice, positions that are not numbers, a filter that keeps no rows) is
   flagged before you can move on. Comma-, tab- and space-separated files are
   all accepted, as `.csv`, `.txt` or `.tsv`.
2. **Outline the plot and check the patches.** Enter the outline of the surveyed
   ground — the x and y ranges as numbers (typed, or stepped 50 m at a time with
   the +/- buttons; a 'from' cannot be stepped past its 'to'), plus any cut-out
   areas or extra blocks. The ranges start
   at the trees' own range, rounded to the nearest 50 — a starting point only;
   check it against the plot's true edge. Cut-out areas and blocks are kept in
   a list: **＋ Add** opens a small editor with the same x/y boxes, the area is
   outlined in orange on the picture as you change it, and **OK** puts it in
   the list (**Cancel** drops it). Each item has ✏️ (edit, same editor) and
   🗑️ (remove) buttons. **Edit as text** shows the same list as lines of
   `xmin, xmax, ymin, ymax`, for pasting many at once. While you list the
   blocks of a plot built from several, the outer range is drawn on its own as
   a frame to place them in. The trees are
   divided into patches and drawn as you type. The grid is
   shaded by how many trees each patch holds, keeping three things clearly apart:
   ground that was **never surveyed** (grey, set aside), patches that *were*
   surveyed and hold **nothing** (hatched purple, labelled — chosen to stay
   distinct for colour-blind readers), and the rest on a pale-to-dark green
   scale. Both of the mistakes described above — an oddly shaped plot treated as
   a rectangle, and patches so small that most come up empty — pass without an
   error and produce normal-looking numbers, but are apparent here at a glance.
   The range of the tree positions is shown as a reminder, and a table lists every patch size that divides your
   plot evenly, with how many species, trees and empty patches each would give.
   Which patch sizes are on offer follows from the outline — every one cuts it
   into whole, square patches — so changing the ranges changes the list; if the
   outline cannot reach 50 m patches (the size the paper uses), the app warns
   you to check the ranges. To try a size that is not listed, type it under
   **Add a patch size**: if it tiles the outline into square patches it joins
   the list, here, in step 3 and in the automatic search; if not, the app says
   why.
3. **Calculate** — pick the patch size from the ones that divide your plot evenly,
   or let it choose; and, if you gave a DBH column, switch weighting by tree size
   on or off (and set the DBH units).
4. **Results** — the fitted numbers, the chart, the CSV download and, behind
   switches, the fit-against-patch-size curve (when the size was chosen
   automatically) and the per-species fits (see section 4), 20 panels per page. Changing
   a sidebar setting here refits straight away; to change the file, the columns
   or the outline, go back to step 1 or 2 — the calculation always uses what
   step 2 showed last.

Every figure (the patch preview, the fit-against-patch-size curve, the
vulnerability chart and the current page of species fits) has **Save PNG** and
**Save PDF** buttons under it; **Save all species (PDF)** writes all pages of
per-species fits to a single file. The
files go wherever your browser saves downloads; to be asked each time, turn on
"Ask where to save each file" in the browser's download settings.

The preview always counts trees, even when a diameter column has been given: on a
map weighted by size, one big tree can hide a patch that is otherwise empty, which
is one of the things this step exists to show.

*Load an example census* loads the demo census *and* fills in the outline it was
surveyed on — the two belong together, since a list of trees says nothing about
which ground was walked. It then goes through the steps like any file of your
own, column choice included; its step 2 shows all three kinds of patch at once. The example is bundled with the app, so it
works from any install, and can be downloaded as CSV to see the exact format
expected.

### Library

`scan_scales` runs the patch-size search and reports it; `compute_vulnerability_census`
takes that choice and goes on to the result.

```python
from vulntool import compute_vulnerability_census, scan_scales, plot_gof_vs_L

scan = scan_scales(trees, footprint=(0, 1000, 0, 500), dbh="dbh")
print(scan.table)              # a row per patch size, with its score
print(scan.recommended_L)      # the size chosen

result, scan = compute_vulnerability_census(trees, footprint=(0, 1000, 0, 500),
                                            dbh="dbh", dbh_units="mm")
print(result.table)            # per-species W_alpha
plot_gof_vs_L(scan)            # score against patch size, with the pass mark
```

`footprint` accepts a rectangle `(xmin, xmax, ymin, ymax)`, a list of them, or an
explicit patch-by-patch mask; `extent=` still works when the plot is a simple
rectangle. Cut-out shapes are `normalize_footprint(..., holes=[...])`. Pass `L=`
to fix the patch size and skip the search, in which case the returned scan is
`None`.

### Command line

```bash
vulnerability census trees.csv --extent 0 1000 0 500 --dbh dbh \
    -o results.csv --gof-plot patch-size.png
```

```
      10 m patches:   5000 patches,  101 species,  misfit 0.3100
      20 m patches:   1250 patches,  101 species,  misfit 0.1900
      50 m patches:    200 patches,  101 species,  misfit 0.1200  <-- fits well
  Using 50 m patches: the smallest size that fits well.
```

Pass `--L 50` to fix the size yourself and skip the search. Use `--dbh-units` if
your diameters are not in millimetres (`mm`, `cm`, `m`, `in`), or `--dbh-scale`
for a factor of your own.

The tree list's column separator is detected from the file; pass `--sep` to set
it yourself (e.g. `--sep '\t'` for tabs, `--sep '\s+'` for runs of spaces).
`--filter COL=V1,V2` keeps only the rows whose `COL` is one of the listed
values; for a ForestGEO table, `--filter status=A` keeps live stems only (see
section 3):

```bash
vulnerability census bci5.txt --extent 0 1000 0 500 --filter status=A \
    --x gx --y gy --species sp --dbh dbh -o results.csv
```

For a plot that is not a rectangle, `--footprint` names one surveyed block and may
be repeated:

```bash
vulnerability census stems.csv \
    --footprint -100 300 0 400 --footprint -200 -100 100 400 \
    --footprint -300 -200 100 200 --footprint 300 400 0 200 \
    --footprint 400 500 0 100 \
    --x gx --y gy --species spcode --dbh dbh --dbh-units cm -o results.csv
```

`--exclude` is the other spelling: it cuts areas out of the outline given by
`--extent`. Here is the same staircase plot in four rectangles rather than five,
giving identical results:

```bash
vulnerability census stems.csv \
    --extent -300 500 0 400 \
    --exclude -300 -100 0 100 --exclude -300 -200 200 400 \
    --exclude 300 400 200 400 --exclude 400 500 100 400 \
    --x gx --y gy --species spcode --dbh dbh --dbh-units cm -o results.csv
```

## 6. Usage guide — data already divided into patches

If your census is already binned into patches, the tool takes those patches as
given. Everything in section 5 about choosing a patch size and describing the plot
outline no longer applies — that work has been done, and the tool cannot check it.
What remains is the species selection and the fit.

One thing does need care here that the census path handles by itself. **Species
are selected from the table you supply.** That is the paper's policy when the
cells are counts of trees. If they are size-weighted — total canopy area, basal
area, biomass — then selecting on that table keeps a different set of species,
because one large tree can carry a species that few individuals would. The
numbers alone do not say which kind of table it is, so in that case you supply a
second table of tree counts for the selection step.

### Browser GUI

The app is the same one described in section 5; choose *Table of patches* as the
input.

```bash
vulnerability gui
```

Upload the table and press *Compute vulnerability*; the results are shown on a
second step, with *← Back* to change the table (and *Next →* to return to
the results while the table and settings are unchanged). *Load an example table*
loads the demo census already divided into 50 m patches, which can also be
downloaded as CSV to see the exact format expected.

If your table is size-weighted, open **My table is size-weighted (e.g. total
canopy area)** and upload the matching table of tree counts there. The species
are then chosen on counts while the vulnerability is still fitted to the
size-weighted figures. The two tables must hold the same species in the same
column order; if they do not, the app says so rather than returning an answer
based on the wrong species.

### Library

```python
from vulntool import compute_vulnerability, plot_vulnerability

result = compute_vulnerability("census.csv")
print(result.table)      # one row per species, ending in W_alpha
print(result.M0)         # how much one patch holds when full

result.to_csv("vulnerability.csv")
plot_vulnerability(result, annotate=5)   # vulnerability against abundance
```

Per-species fits:

```python
from vulntool import plot_species_fits, save_species_fits_pdf
from vulntool.plotting import species_fit_order

plot_species_fits(result, result.table.index[:20])   # first 20, descending abundance
plot_species_fits(result, species_fit_order(result, worst_first=True)[:10])
save_species_fits_pdf(result, "fits.pdf")             # all species, 25 per page
```

Results are in `result.table`, fit quality in `result.cvm`, patch capacity in
`result.M0`. The pairing to look at is `p_alpha` against `W_alpha`, which is what
`plot_vulnerability` draws. Species at the left-hand edge are the rare ones; how
high or low they sit says whether being rare is actually costing them.

To select species on one table while fitting another, pass the second table as
`selection_source`. This is how the paper's own analysis is reproduced: species
kept on tree counts, abundance measured as canopy area.

```python
compute_vulnerability("crown.csv", selection_source="tree_counts.csv")
```

Omitting it means species are selected from `"crown.csv"` itself, which is a
different — and, if the cells are size-weighted, a less appropriate — set.

### Command line

```bash
vulnerability fit census.csv -o results.csv --plot chart.png --annotate 5
```

This writes the per-species table to `results.csv`, and the chart of
vulnerability against abundance to `chart.png`, labelling the five most
vulnerable species. Without `-o` the table is printed instead.

`--selection tree_counts.csv` is the command-line spelling of `selection_source`
above. `--fits-pdf fits.pdf` additionally writes the per-species fits (section 4),
ordered by descending abundance; the option is also available for
`vulnerability census`.

## 7. How the vulnerability extraction works

For a thorough description of the theory underlying the tool's working, we refer to the paper.

Here, a brief operational guide is provided. The information flow is:

```
table of patches ─▶ pick the species ─▶ fit how each is spread ─▶ vulnerability
```

- **Which species.** The commonest are kept, down to where they account for 95%
  of all individuals, and any species missing from more than 90% of patches is
  dropped. A species present in a handful of patches gives the model almost
  nothing to work with, however many trees it has there. Both thresholds are
  adjustable (`--frac-keep`, `--max-zero-frac`, or the sliders in the app).
- **How full a patch can get** is fitted from your data and reported. You never
  set it, and it is what everything else is measured against.
- **How each species is spread** is fitted only over the patches where it is
  actually present. The fitted distribution is continuous, so it never predicts a
  value of exactly zero; scoring it against absences would judge it on something
  it does not claim to describe.

### Bad input is refused, not fitted

When a census is malformed, the danger is not a crash — it is a confident wrong
answer. So blank or non-numeric cells, negative abundances, repeated species
names, and tables with too few patches (usually a sign the file is the wrong way
round) all stop with an error naming what is wrong and what to do about it. If
blanks in your file genuinely mean "none here", `--allow-nan` says so.

## 8. The test suite

The test suite checks that this reproduces the paper's stored result to
within one part in a million.

It is checked against the original code, end to end from raw tree lists, against two
datasets of the original publication (BCI and Michigan),
one with a rectangular plot and one with an irregular staircase-shaped plot:

| Check | Result |
|---|---|
| Tree counts per patch (rectangular plot, 50 m, 200 patches) | **exactly** equal |
| Canopy area per patch (rectangular plot) | equal to 15 decimal places |
| Which patches are in the plot, at all 5 sizes (irregular plot) | **identical**, patch for patch |
| Canopy area per patch (irregular plot, 92 of 128 patches) | equal to 14 decimal places |
| The patch sizes worked out from the shape (irregular plot) | matches the paper's own list |
| The fitted numbers, including `W_alpha` (rectangular plot) | equal to one part in a million |

Every one of those checks needs the real census data, available online.
If you need to run these tests, point `VULNTOOL_CAPSULE_DATA` at a copy of the
paper's code archive:

```bash
VULNTOOL_CAPSULE_DATA=/path/to/capsule pytest    # 140 passed
pytest                                           # 116 passed, 24 skipped
```

These tests skip when it is absent, which is how the automated builds run. What
still runs there is a check on the bundled demo census
(`tests/test_golden_synthetic.py`), which records the numbers this code currently
produces so that an accidental change to the maths shows up somewhere. That is a
record of what the code *does*, not evidence that it is right. The automated builds 
cover Python 3.10 to 3.12 and a clean install from scratch.


### Note on the spatial scale selection of the original publication

The rule above judges each plot on its own data. On the paper's three sites,
with the sizes offered by default, it picks 50 m for BCI and Pasoh and 20 m for
Michigan; the paper fixes 50 m across all three, so that the sites can be
compared with each other. Sizes between those on offer can pass too — BCI
already fits well at 31.25 m (the paper's "L=31"), which can be added by hand.

## 9. Citing this tool

If you use this tool in work you publish, please cite the paper it implements:

> Davide Bernardi, Giorgio Nicoletti, Prajwal Padmanabha, Samir Suweis, Sandro
> Azaele, Simon A. Levin, Andrea Rinaldo and Amos Maritan (2026). Dispersal
> diversity buffers species vulnerability to local extinction. Nature Ecology &
> Evolution. https://doi.org/10.1038/s41559-026-03195-y

```bibtex
@article{bernardi2026vulnerability,
  title = {Dispersal diversity buffers species vulnerability to local extinction},
  author = {Davide Bernardi and Giorgio Nicoletti and Prajwal Padmanabha and Samir Suweis and Sandro Azaele and Simon A. Levin and Andrea Rinaldo and Amos Maritan},
  year = {2026},
  journal = {Nature Ecology \& Evolution},
  doi = {10.1038/s41559-026-03195-y},
  url = {https://doi.org/10.1038/s41559-026-03195-y},
}
```

The article is at https://www.nature.com/articles/s41559-026-03195-y; its
Supplementary Information shows the same per-species fits for the paper's
three forests.

The same reference is in `CITATION.cff` at the repository root, which GitHub
turns into a *Cite this repository* button, and is shown in the app alongside
the results.

## 10. License

MIT — see `LICENSE`.
