from importlib.metadata import metadata
from datetime import datetime
from pathlib import Path


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))

info = metadata("opendvp")  # replace with your package name as installed via pip
project = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}"
release = info["Version"]


project = 'opendvp'
copyright = '2025, Jose Nimo'
author = 'Jose Nimo'
release = '0.2.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # for Google-style or NumPy-style docstrings
    'sphinx_autodoc_typehints',
    'nbsphinx',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

# Autodoc options

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}