# vulntool

This Python package is a user-friendly implementation of the species vulnerability
metric introduced in:

*"Dispersal diversity buffers species vulnerability to local extinction"*,
*Nature Ecology & Evolution* (2026),
[doi:10.1038/s41559-026-03195-y](https://www.nature.com/articles/s41559-026-03195-y).

Given a single census, this package measures the metric automatically and reports it per
species. If you use it in a publication, please cite the paper above.

Stem tables in the ForestGEO (CTFS) format, such as the Barro Colorado Island census, can
be loaded directly. The GUI tries to match all relevant columns automatically,
including leaving out dead stems, and also handles non-rectangular plot shapes.

This README only covers the basics; see [GUIDE.md](GUIDE.md) for more details.

## Install

Three ways to use it:

1. A graphical user interface (GUI) based on Streamlit.
2. A command line interface (CLI)
3. Importing the library directly into your own analysis scripts 

The full install command is:

```bash
pip install "vulntool[gui] @ git+https://github.com/davideb321/vulntool"
```

Then launch it with `vulnerability gui` (a local browser app), or use
`vulnerability` on the command line, or `import vulntool` in your own scripts
— all three come with any install.

Leave off `[gui]` to skip the browser app and its Streamlit dependency (a few hundred MB) 
if you only need the CLI or the library.
 **Use the double quotes** — unquoted, `[gui]` is a wildcard
pattern in zsh and PowerShell. See [GUIDE.md](GUIDE.md) for installing from a
clone, pinning a version, and installing without git.

## Update

Re-run the install command with `--upgrade`:

```bash
pip install --upgrade "vulntool[gui] @ git+https://github.com/davideb321/vulntool"
```

The GUI's title bar shows the installed version, so you can check you're on
the latest one after upgrading.

## Where things are

`pip` puts the code in your Python environment's `site-packages` folder, like
any other package; you never need to open it. To see the exact path:

```bash
python -c "import vulntool, os; print(os.path.dirname(vulntool.__file__))"
```

The two demo files the GUI loads with its "example" buttons
(`demo_census.csv`, `demo_patch_table.csv`) live in that folder. The larger
[`examples/`](examples/) directory in this repository (a real census, a
notebook) is **not** installed — download it from GitHub or clone the repo.

Nothing is written to disk unless you ask: the GUI has a download button on
the results page, and the command line prints the table unless you pass
`-o results.csv`.

## Input formats

Two kinds of input are supported:
- a **list of individual trees**, in which each row is a single tree,
and columns must indicate at least the x, y coordinates in m, a species label,
and optionally a trunk diameter and a status code.
The tool guides the user through the process of divides the plot into patches itself,
 and can compute the optimal patch size automatically.
- a **table already divided into patches**
(species along the top, a patch label first, one row per patch) — the same census
with that division already done.

## Output

`compute_vulnerability(...)` returns one row per species, ending in
`W_alpha` — the vulnerability index, more negative meaning better protected (less vulnerable)
— alongside `abundance`, `p_alpha` (how common the species is), and the
fitted spread parameters `delta`/`betabar`. See [GUIDE.md](GUIDE.md) for the
full column glossary and how to plot the result.

## License

MIT — see `LICENSE`.
