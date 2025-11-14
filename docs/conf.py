# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# import module for docs creation
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "TorchSurv"
copyright = "2024, Novartis Pharma AG"
author = "Thibaud Coroller, Mélodie Monod, Peter Krusche, Qian Cao"
release = "0.1.5"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_math_dollar",
]


# templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# path to citations
bibtex_bibfiles = ["source/references.bib"]
suppress_warnings = ["bibtex"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "_static/logo_firecamp.png"
templates_path = ["_templates"]
html_css_files = ["custom.css"]
html_theme_options = {
    "announcement": (
        "🎉 We are part of the "
        "<a href='https://cdrh-rst.fda.gov/torchsurv-deep-learning-tools-survival-analysis'> FDA's Regulatory Science Tools Catalog!</a> 🎉"
    ),
    "repository_url": "https://github.com/Novartis/torchsurv",
    "repository_branch": "main",
    "use_repository_button": True,
}

# latex_engine = 'xeltex'
latex_elements = {
    "preamble": r"""
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc} % The default since 2018
\DeclareUnicodeCharacter{200B}{{\hskip 0pt}}
\DeclareUnicodeCharacter{2223}{{\hskip 0pt}}
""",
    "printindex": r"\footnotesize\raggedright\printindex",
}
latex_show_urls = "footnote"

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
    "special-members": "__call__, __init__",
    "private-members": False,
    "imported-members": False,
    "recurse": True,
    "collapse": False,
}

autosummary_generate = True
autosummary_imported_members = False

# Additional autodoc configuration
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Force autosummary to respect source order
autosummary_context = {}
add_module_names = False
