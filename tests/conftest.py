"""Shared fixtures.

The golden tests compare against the published analysis' own output, which
means they need the reproducibility capsule's raw census data. That data is
separately licensed (BCI, and Michigan under CC BY-NC 4.0),
so it is **not** redistributed here: point ``VULNTOOL_CAPSULE_DATA`` at a local
capsule checkout to run them, and they skip cleanly otherwise.

    VULNTOOL_CAPSULE_DATA=/path/to/capsule pytest
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

#: Relative locations to probe when the environment variable is unset.
_FALLBACKS = ("../capsule", "../../capsule")


def _find_capsule():
    # An explicitly set variable is taken at its word: falling back to a probe
    # would quietly test something other than what the user pointed at.
    env = os.environ.get("VULNTOOL_CAPSULE_DATA")
    if env:
        path = Path(env)
        return path.resolve() if (path / "raw_data").is_dir() else None

    here = Path(__file__).resolve().parent.parent
    for path in (here / rel for rel in _FALLBACKS):
        if (path / "raw_data").is_dir():
            return path.resolve()
    return None


CAPSULE = _find_capsule()

requires_capsule = pytest.mark.skipif(
    CAPSULE is None,
    reason="capsule census data not found; set VULNTOOL_CAPSULE_DATA to a capsule checkout",
)


@pytest.fixture(scope="session")
def capsule():
    """Path to a capsule checkout containing ``raw_data/``."""
    if CAPSULE is None:
        pytest.skip("set VULNTOOL_CAPSULE_DATA to a capsule checkout")
    return CAPSULE


def align_to_reference(data, species, binned, ref_xy, ref_species):
    """Reorder a computed matrix onto the reference file's own layout.

    The original coarse-graining scripts and vulntool disagree on two arbitrary
    conventions: those scripts walk patches with y varying fastest and order
    species by first appearance, while vulntool walks x fastest and sorts
    species. Neither affects the fit (patches
    and species are exchangeable), so aligning by *label* — patch centre
    coordinate and species name — is what makes the comparison meaningful rather
    than a test of bookkeeping.
    """
    import numpy as np
    from vulntool.footprint import grid_shape, patch_centers

    nx, ny, _, _ = binned.grid
    cx, cy = patch_centers(binned.footprint, nx, ny)
    pos = {xy: i for i, xy in enumerate(zip(cx[binned.mask], cy[binned.mask]))}
    order = [pos[(rx, ry)] for rx, ry in ref_xy]

    by_name = {n: i for i, n in enumerate(species)}
    reordered = data[:, order]
    n_patches = reordered.shape[1]
    return np.array([
        reordered[by_name[n]] if n in by_name else np.zeros(n_patches)
        for n in ref_species
    ])
