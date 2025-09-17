# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from importlib.metadata import version as package_version

project = "DeepSDFStruct"
copyright = "2025, Michael Kofler"
author = "Michael Kofler"
release = str(package_version("DeepSDFStruct"))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# -- Extensions --------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]


autosummary_context = {"skipmethods": ["__init__"]}


autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "autosummary": True,
}

autodoc_typehints_format = "fully-qualified"  # show full package.module.Class
autodoc_typehints_description_target = "all"  # document args + return types
autodoc_typehints = "both"


html_theme_options = {"collapse_navigation": False, "navigation_depth": 4}
