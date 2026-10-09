"""Entry script for ``streamlit run`` (see ``vulnerability gui``).

Streamlit executes its target as a top-level script, not as a package member, so
a module full of relative imports cannot be pointed at directly. This shim uses
an absolute import instead, which lets :mod:`vulntool.gui` stay an ordinary,
importable, testable module.
"""
from vulntool.gui import run

run()
