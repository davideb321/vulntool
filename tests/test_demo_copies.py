"""The demo CSVs exist twice; this is what keeps the two copies identical.

``src/vulntool/`` holds the copies that ship. Every code path reads those —
:func:`vulntool.load_demo_census`, :func:`vulntool.load_demo_patch_table` and
the GUI's example buttons all resolve them through ``importlib.resources``, so
they work from a plain ``pip install`` where no ``examples/`` directory exists.

``examples/`` holds a second copy that ships nowhere: not in the wheel, not in
the sdist. It is there for someone reading the repo on GitHub, and because the
guide's command-line example needs a real path on disk:

    vulnerability census examples/demo_census.csv ...

Nothing else would notice the two drifting apart. The rest of the suite goes
through ``load_demo_*``, i.e. the shipped copy, so a regenerated ``examples/``
file would leave the guide documenting one thing and the tool loading another
with every test still green. Hence a byte comparison, which is the whole
guarantee: the files are a pure function of the generator's seed, so identical
inputs must give identical bytes, not merely equivalent data.

``dev/simulate_demo_data.py`` writes both copies in one pass, which is what
makes them agree in the first place. It is deliberately not part of this suite
(it is gitignored, and regenerating data is not a test), so this check is the
only thing in CI standing between the two copies and silent divergence.
"""
import importlib.resources
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"

#: The files that exist in both places.
DEMO_FILES = ["demo_census.csv", "demo_patch_table.csv"]


@pytest.mark.parametrize("name", DEMO_FILES)
def test_the_examples_copy_matches_the_shipped_copy(name):
    example = EXAMPLES / name
    if not example.is_file():
        # An installed distribution and the sdist both omit examples/ on
        # purpose, so the comparison is only meaningful against a checkout.
        # Same reasoning as GUIDE.md in test_citation.py.
        pytest.skip("examples/ is not part of an installed distribution")

    shipped = importlib.resources.files("vulntool") / name

    assert example.read_bytes() == shipped.read_bytes(), (
        f"examples/{name} and src/vulntool/{name} have drifted apart. "
        "The tool loads the src/vulntool copy while the guide points readers at "
        "the examples one, so they must be identical. Regenerate both with "
        "dev/simulate_demo_data.py rather than editing either by hand."
    )
