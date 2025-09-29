import os
import sys

import sphinx_rtd_theme

# Add the path to the folder containing your Python source code files
sys.path.insert(
    0,
    os.path.abspath("../src")
)

# -- Project information -----------------------------------------------------

project = "project_NLP8"
copyright = (
    "2025, Alexandru Mic, Maria Salop, Anastasiia Mokhonko, "
    "Nikita Orlovs, Artjom Musaelans"
)
copyright = (
    "2025, Alexandru Mic, Maria Salop, Anastasiia Mokhonko, "
    "Nikita Orlovs, Artjom Musaelans"
)
author = (
    "Alexandru Mic, Maria Salop, Anastasiia Mokhonko, "
    "Nikita Orlovs, Artjom Musaelans"
    "Alexandru Mic, Maria Salop, Anastasiia Mokhonko, "
    "Nikita Orlovs, Artjom Musaelans"
)
release = "notspecified"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]

templates_path = ["_templates"]
exclude_patterns = []
language = "en"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# Ensure theme path is found
html_static_path = ["_static"]
