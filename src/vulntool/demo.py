"""The demo census: a small forest to try the tool on.

Two example files come bundled with every install —
``demo_census.csv`` (a raw list of stems) and ``demo_patch_table.csv`` (the
same census already divided into patches) — a synthetic 20 ha plot of 30 000
stems across 40 species. Species are simulated with a **Thomas process**
(cluster centres scattered over the plot, stems scattered around each centre,
with a cluster width that varies by species), so the community spans a real
range of patchiness from tight clumps to near-uniform, rather than the
uniform scatter a plain random placement would give — the pattern this
package measures needs to actually be there to make a useful demo.

The stem list is laid out like a ForestGEO census table: besides the 30 000
live stems (``status`` ``A``) it lists about 3 000 dead (``D``) and missing
(``M``) ones, without a DBH, so that leaving them out — which real census
files require — is part of the demo too. Keeping ``status == "A"`` gives the
census every result is computed from; the patch table holds those stems only.

The plot is a 600 x 400 m rectangle with its north-east quarter never
surveyed (a hole) and a small treeless clearing inside the surveyed area, so
all three patch states — unsurveyed, surveyed-but-empty, and occupied — show
up when previewing it.

    >>> from vulntool import load_demo_census, DEMO_PLOT
    >>> stems = load_demo_census()
    >>> stems.columns.tolist()
    ['x', 'y', 'species', 'dbh', 'status']
"""
from __future__ import annotations

import importlib.resources
from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from .footprint import Footprint, normalize_footprint
from .io import load_matrix

_PACKAGE_FILES = importlib.resources.files(__package__)


@dataclass(frozen=True)
class DemoPlot:
    """The geometry of the shipped example plot.

    A 600 x 400 m plot with its north-east quarter never surveyed (a ``hole``)
    and a small unforested clearing inside the surveyed area (``clearings``).
    The two are different things and the distinction matters: patches in the
    hole are *dropped*, patches in the clearing are *kept and empty*. Both
    states appear in the demo, because both occur in real censuses and confusing
    them is the mistake this package's footprint handling exists to prevent.
    """

    extent: Tuple[float, float, float, float] = (0.0, 600.0, 0.0, 400.0)
    #: Unsurveyed block, removed from the plot outline.
    holes: Tuple[Tuple[float, float, float, float], ...] = ((400.0, 600.0, 200.0, 400.0),)
    #: Surveyed but treeless (a swamp, a rock outcrop, a light gap).
    clearings: Tuple[Tuple[float, float, float, float], ...] = ((250.0, 300.0, 50.0, 150.0),)

    @property
    def footprint(self) -> Footprint:
        return normalize_footprint(self.extent, holes=self.holes)


#: The example plot used by the tutorial, the GUI demo button and the tests.
DEMO_PLOT = DemoPlot()


def demo_census_bytes() -> bytes:
    """The bundled demo census, as the raw bytes of its CSV file."""
    return (_PACKAGE_FILES / "demo_census.csv").read_bytes()


def load_demo_census() -> pd.DataFrame:
    """The bundled demo census as a stem list: one row per stem.

    Columns ``x``, ``y`` (metres), ``species``, ``dbh`` (millimetres) and
    ``status``, in the ForestGEO convention: ``A`` alive, ``D`` dead, ``M``
    missing. Only live stems should be analysed::

        stems = keep_rows(load_demo_census(), "status", ["A"])

    (:func:`vulntool.io.keep_rows`); the dead and missing ones have no DBH.
    """
    with (_PACKAGE_FILES / "demo_census.csv").open("rb") as f:
        return pd.read_csv(f)


def load_demo_patch_table() -> pd.DataFrame:
    """The bundled demo census already divided into patches, in table format.

    One row per patch, species along the top, a patch label first — what
    :func:`vulntool.compute_vulnerability` reads. Useful for showing that
    format to someone who has never seen it, and for exercising the table
    path with no file to hand. Patches outside the plot outline are absent,
    exactly as they would be for a real irregular plot.
    """
    data, species, patches = load_matrix((_PACKAGE_FILES / "demo_patch_table.csv").open("rb"))
    table = pd.DataFrame(data.T, columns=species)
    table.insert(0, "patch", patches)
    return table
