"""The citation appears in four places; this is what keeps them in step.

``src/vulntool/citation.py`` is the source of truth. The app and the CLI import
it, so those two cannot drift. ``CITATION.cff``, ``README.md`` and ``GUIDE.md``
carry the same reference as plain text, and the tests below check them against
the module — so updating the paper's details and forgetting a file fails here
rather than shipping a half-updated citation.

Substring checks, not a YAML parse: catching drift needs no parser, and pulling
in a YAML dependency for one file would not be worth it.
"""
from pathlib import Path

import pytest

from vulntool import citation

ROOT = Path(__file__).resolve().parent.parent

#: Files that repeat the citation, and must therefore be updated with it.
TEXT_FILES = ["CITATION.cff", "README.md", "GUIDE.md"]


def _read(name):
    """Contents of a repo-root file, or a skip when it is not shipped.

    An installed distribution has no GUIDE.md, and an sdist need not carry
    CITATION.cff — the check is meaningful only against a checkout.
    """
    path = ROOT / name
    if not path.is_file():
        pytest.skip(f"{name} is not part of an installed distribution")
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", TEXT_FILES)
def test_the_paper_url_matches_the_citation_module(name):
    assert citation.URL in _read(name), (
        f"{name} does not carry {citation.URL}. Update it to match "
        "src/vulntool/citation.py."
    )


@pytest.mark.parametrize("name", TEXT_FILES)
def test_the_title_matches_the_citation_module(name):
    assert citation.TITLE in _read(name), (
        f"{name} does not carry the paper's title as spelled in "
        "src/vulntool/citation.py."
    )


def test_the_cff_lists_every_author_in_order():
    """The .cff is the machine-readable copy, so its author list has to be whole."""
    text = _read("CITATION.cff")
    positions = []
    for name in citation.AUTHORS:
        family = name.split()[-1]
        assert f"family-names: {family}" in text, f"{name} is missing from CITATION.cff"
        positions.append(text.index(f"family-names: {family}"))
    assert positions == sorted(positions), "CITATION.cff lists the authors out of order"


def test_bibtex_is_a_plausible_entry():
    bib = citation.BIBTEX
    assert bib.startswith("@")
    assert bib.rstrip().endswith("}")
    assert citation.TITLE in bib
    assert all(name in bib for name in citation.AUTHORS)
    # Balanced braces: an unbalanced entry breaks silently on paste into LaTeX.
    assert bib.count("{") == bib.count("}")
    # So does a bare &, as in the journal's name.
    assert "&" not in bib.replace(r"\&", "")


def test_the_preprint_and_published_forms_stay_consistent(monkeypatch):
    """Setting DOI switches every rendering over; nothing keeps saying arXiv."""
    monkeypatch.setattr(citation, "DOI", "10.1038/s41559-000-00000-0")
    monkeypatch.setattr(citation, "VENUE", "Nature Ecology & Evolution")

    text, bib = citation.citation_text(), citation.bibtex()
    assert "10.1038/s41559-000-00000-0" in text
    assert citation.ARXIV_ID not in text
    assert bib.startswith("@article")
    assert "archivePrefix" not in bib


def test_the_request_is_one_short_sentence():
    """It is rendered in the results expander and an argparse epilog; keep it small."""
    assert len(citation.REQUEST) < 160
    assert citation.REQUEST.count(".") == 1


def test_link_is_the_journal_doi_and_the_arxiv_doi_without_one(monkeypatch):
    assert citation.link() == f"https://doi.org/{citation.DOI}"
    monkeypatch.setattr(citation, "DOI", None)
    assert citation.link() == f"https://doi.org/10.48550/arXiv.{citation.ARXIV_ID}"


def test_the_text_files_carry_the_doi():
    for name in TEXT_FILES:
        assert citation.DOI in _read(name), f"{name} does not carry the DOI"
