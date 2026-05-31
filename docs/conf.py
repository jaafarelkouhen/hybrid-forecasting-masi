"""Sphinx configuration for MASI Hybrid Forecasting docs."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


project = "MASI Hybrid Forecasting"
author = "Jaafar El Kouhen"
copyright = f"{datetime.now().year}, {author}"
release = "1.0.0"
version = "1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
    "tasklist",
    "fieldlist",
    "html_image",
    "substitution",
]
myst_heading_anchors = 3

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

master_doc = "home"
language = "en"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "references/**",
    "README.md",
    "README_legacy.md",
]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "titles_only": False,
}
html_title = "MASI Hybrid Forecasting"
html_short_title = "MASI Forecasting"

html_static_path = ["_static"] if (DOCS_DIR / "_static").exists() else []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"

suppress_warnings = ["myst.header"]

on_rtd = os.environ.get("READTHEDOCS") == "True"
if on_rtd:
    html_context = {
        "display_github": True,
        "github_user": "jaafarelkouhen",
        "github_repo": "masi-hybrid-forecasting",
        "github_version": "main",
        "conf_py_path": "/docs/",
    }
