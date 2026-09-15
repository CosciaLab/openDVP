from importlib.metadata import metadata
from pathlib import Path
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

add_module_names = False

html_context = {
    "display_github": True,
    "github_user": "CosciaLab",
    "github_repo": "openDVP",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_theme_options = {
    "navigation_depth": -1,
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
napoleon_google_docstring = False
napoleon_numpy_docstring = True
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
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    # placeholder notebooks with no content yet
    "Tutorials/T4_Segmask_to_shapes.ipynb",
    "Tutorials/T5_Thresholding_tutorial.ipynb",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
# html_css_files = ["css/custom.css"]
html_title = project
# html_logo = '_static/images/logo.png'
# html_favicon = '_static/images/logo.png'

pygments_style = "default"

nitpick_ignore = []

# numpydoc type strings are prose, not just types: napoleon splits them on commas and sphinx
# then tries to resolve each fragment as a class. Ignore the fragments that are never objects.
nitpick_ignore_regex = [
    ("py:class", r"^optional$"),
    ("py:class", r"^default.*"),
    ("py:class", r"^callable$"),
    ("py:class", r"^array-like$"),
    ("py:class", r"^[\d.]+%?$"),
    ("py:class", r"^[\"'{].*"),
    ("py:class", r".*[\"'}]$"),
    # libpysal publishes no objects.inv to intersphinx against
    ("py:class", r"^libpysal\..*"),
    # private paths sphinx resolves annotations to; anndata only publishes anndata.AnnData
    ("py:class", r"^anndata\._core\..*"),
]
