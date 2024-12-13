# pylint: skip-file

project = "Projektpraktikum I (WiSe2024)"
copyright = "2024, P. Merz, M. van Straten"
author = "P. Merz, M. van Straten"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "myst_parser",
]
autosummary_generate = True

import sys
from pathlib import Path

sys.path.insert(0, str(Path("..", "src/").resolve()))
