"""vulntool — how exposed each forest species is to local extinction.

This Python package is a user-friendly implementation of the species vulnerability
metric introduced in:

*"Dispersal diversity buffers species vulnerability to local extinction"*,
*Nature Ecology & Evolution* (2026),
[doi:10.1038/s41559-026-03195-y](https://www.nature.com/articles/s41559-026-03195-y).

Given a single census, this package measures the metric automatically and reports it per
species. If you use it in a publication, please cite the paper above.

Quick start
-----------
    >>> from vulntool import compute_vulnerability
    >>> result = compute_vulnerability("census.csv")
    >>> result.table          # one row per species, ending in W_alpha
    >>> result.M0             # how much one patch holds when full

If you have a list of individual trees rather than a table of patches, use
:func:`vulntool.compute_vulnerability_census`, which divides the plot into
patches and works out a sensible patch size for you. :func:`load_demo_census`
loads a bundled demo census to try either on.
"""
from .pipeline import (
    compute_vulnerability,
    compute_vulnerability_census,
    scan_scales,
    VulnerabilityResult,
    ScaleScan,
)
from .theory import w_alpha, compute_derived, get_xs_from_data
from .selection import select_species
from .gof import cvm_gamma_xs, median_cvm, CVM_REFERENCE
from .io import (
    load_matrix, load_bare_matrix, bin_census, as_matrix,
    census_matrices, default_scales, crown_from_dbh, BinnedCensus,
)
from .footprint import Footprint, normalize_footprint, admissible_scales, check_scale
from .validate import validate_matrix
from .demo import load_demo_census, load_demo_patch_table, DemoPlot, DEMO_PLOT
from .citation import CITATION_TEXT, BIBTEX

__version__ = "0.3.2"

__all__ = [
    "compute_vulnerability",
    "compute_vulnerability_census",
    "scan_scales",
    "VulnerabilityResult",
    "ScaleScan",
    "w_alpha",
    "compute_derived",
    "get_xs_from_data",
    "select_species",
    "cvm_gamma_xs",
    "median_cvm",
    "CVM_REFERENCE",
    "load_matrix",
    "load_bare_matrix",
    "bin_census",
    "census_matrices",
    "default_scales",
    "crown_from_dbh",
    "as_matrix",
    "BinnedCensus",
    "Footprint",
    "normalize_footprint",
    "admissible_scales",
    "check_scale",
    "validate_matrix",
    "load_demo_census",
    "load_demo_patch_table",
    "DemoPlot",
    "DEMO_PLOT",
    "plot_vulnerability",
    "plot_gof_vs_L",
    "plot_species_fits",
    "save_species_fits_pdf",
    "CITATION_TEXT",
    "BIBTEX",
]


def plot_vulnerability(*args, **kwargs):
    """Lazy wrapper so importing vulntool doesn't require matplotlib."""
    from .plotting import plot_vulnerability as _p
    return _p(*args, **kwargs)


def plot_gof_vs_L(*args, **kwargs):
    """Lazy wrapper so importing vulntool doesn't require matplotlib."""
    from .plotting import plot_gof_vs_L as _p
    return _p(*args, **kwargs)


def plot_species_fits(*args, **kwargs):
    """Lazy wrapper so importing vulntool doesn't require matplotlib."""
    from .plotting import plot_species_fits as _p
    return _p(*args, **kwargs)


def save_species_fits_pdf(*args, **kwargs):
    """Lazy wrapper so importing vulntool doesn't require matplotlib."""
    from .plotting import save_species_fits_pdf as _p
    return _p(*args, **kwargs)
