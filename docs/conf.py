from importlib.metadata import metadata
from datetime import datetime
from pathlib import Path
import os
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))

# -- Project information -----------------------------------------------------

info = metadata("opendvp")
project = info["Name"]
author = "Jose Nimo"
# author = info["Author"]
# copyright = f"{datetime.now():%Y}, {author}"
release = info["Version"]

# urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
# repository_url = urls["Source"]
repository_url = "https://github.com/CosciaLab/openDVP"

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]
nitpicky = True
needs_sphinx = "4.0"

html_context = {
    "display_github": True,
    "github_user": "CosciaLab",
    "github_repo": "openDVP",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx_tabs.tabs",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

autosummary_generate = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = True #changed this
napoleon_numpy_docstring = False #changed this
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
myst_heading_anchors = 6
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
    # ".md" : "markdown",
    # ".md": "myst",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
# html_css_files = ["css/custom.css"]
html_title = project
# html_logo = '_static/images/logo.png'
# html_favicon = '_static/images/logo.png'

pygments_style = "default"

nitpick_ignore = []