# Configuration file for the Sphinx documentation builder.
import sys
from pathlib import Path

# Insert source path to allow autodoc to find the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# -- Project information -----------------------------------------------------
project = 'jc-sim-oqp'
copyright = '2026, Ioan Sturzu'  # noqa: A001
author = 'Ioan Sturzu'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # Generate docs from docstrings
    'sphinx.ext.napoleon',      # Support Google/NumPy style docstrings
    'sphinx.ext.viewcode',      # Add links to highlighted source code
    'sphinx.ext.mathjax',       # Render math via MathJax
    'myst_parser',              # Parse Markdown
    'sphinx_copybutton',        # Add copy button to code blocks
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_title = "JC-Sim-OQP Documentation"

html_theme_options = {
    "repository_url": "https://github.com/isturzu/jc-sim-oqp",
    "use_repository_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
    "home_page_in_toc": True,
}

# -- MyST Configuration ------------------------------------------------------
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
]
