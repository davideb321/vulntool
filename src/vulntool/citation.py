"""How to cite the paper this tool implements.

**This module is the single source of truth for the citation.** The app and the
command line read it directly, so they can never fall out of step. Three other
places carry the same reference as plain text — ``CITATION.cff``, ``README.md``
and ``GUIDE.md`` — and ``tests/test_citation.py`` checks them against the values
here, so a half-finished update fails the test suite instead of shipping.

When the paper appears in a journal, edit the values below, then run ``pytest``:
it will name every file still carrying the preprint.
"""
from __future__ import annotations

#: Authors, in the order they appear on the paper.
AUTHORS = [
    "Davide Bernardi",
    "Giorgio Nicoletti",
    "Prajwal Padmanabha",
    "Samir Suweis",
    "Sandro Azaele",
    "Simon A. Levin",
    "Andrea Rinaldo",
    "Amos Maritan",
]

TITLE = "Dispersal diversity buffers species vulnerability to local extinction"
YEAR = 2026

#: Where the paper lives: the journal article. The preprint's identifier is
#: kept for :func:`link` and :func:`bibtex` should :data:`DOI` ever be unset.
ARXIV_ID = "2604.26589"
URL = "https://www.nature.com/articles/s41559-026-03195-y"
VENUE = "Nature Ecology & Evolution"

#: The article's DOI; ``None`` would mean "not published yet" and keep the
#: DOI line out of every rendering below.
DOI = "10.1038/s41559-026-03195-y"


def _authors_text(sep=", ", last=" and "):
    if len(AUTHORS) == 1:
        return AUTHORS[0]
    return sep.join(AUTHORS[:-1]) + last + AUTHORS[-1]


def link():
    """Where a click on the title should go: the journal DOI once there is one,
    until then the DOI arXiv assigns every preprint."""
    if DOI:
        return f"https://doi.org/{DOI}"
    return f"https://doi.org/10.48550/arXiv.{ARXIV_ID}"


def citation_text():
    """The citation as one line of prose, for printing or displaying."""
    parts = [f"{_authors_text()} ({YEAR}). {TITLE}. {VENUE}."]
    if DOI:
        parts.append(f"https://doi.org/{DOI}")
    else:
        parts.append(URL)
    return " ".join(parts)


def bibtex():
    """The citation as a BibTeX entry."""
    authors = " and ".join(AUTHORS)
    kind, key = ("misc", "bernardi2026vulnerability")
    fields = [
        ("title", TITLE),
        ("author", authors),
        ("year", str(YEAR)),
    ]
    if DOI:
        kind = "article"
        # A bare & is an alignment character in LaTeX and breaks the entry.
        fields += [("journal", VENUE.replace("&", r"\&")), ("doi", DOI), ("url", f"https://doi.org/{DOI}")]
    else:
        fields += [("eprint", ARXIV_ID), ("archivePrefix", "arXiv"), ("url", URL)]

    body = "\n".join(f"  {name} = {{{value}}}," for name, value in fields)
    return f"@{kind}{{{key},\n{body}\n}}"


#: One short sentence, for places with no room for the full reference.
REQUEST = (
    "If you use this tool in work you publish, please cite the paper it "
    "implements."
)

CITATION_TEXT = citation_text()
LINK = link()
BIBTEX = bibtex()
