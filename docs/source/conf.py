# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#


import os
import sys
from typing import Any, Dict

import sphinx_fontawesome  # noqa: F401
from sphinx.ext.autodoc import between

# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "NeuralProphet"
copyright = "2024, Oskar Triebe"
author = "Oskar Triebe"
version = "1.0.0"
release = "1.0.0rc8"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx_fontawesome",
]
html_sourcelink_suffix = ""

# Here to describe what format of files are parsed
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# Add any paths that contain templates here, relative to this directory.
# Note: in use for custom sidebar and landing page
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_favicon = "images/np_favicon.png"
html_logo = "images/np_highres_docs.svg"
# html_logo = "images/logo.png"
font_stack = "-apple-system,'system-ui','Segoe UI',Helvetica,Arial,sans-serif,'Apple Color Emoji','Segoe UI Emoji'"
font_stack_mono = "'SFMono-Regular',Menlo,Consolas,Monaco,Liberation Mono,Lucida Console,monospace"
html_theme_options: Dict[str, Any] = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "light_css_variables": {
        "font-stack": font_stack,
        "font-stack--monospace": font_stack_mono,
    },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static", "images/np_highres.svg"]

# html_sidebars = { '**': [
#     "_templates/sidebar/brand.html",
#     "sidebar/search.html",
#     "sidebar/scroll-start.html",
#     "sidebar/navigation.html",
#     "sidebar/ethical-ads.html",
#     "sidebar/scroll-end.html",
# ] }

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg'}",
    "--InlineBackend.rc=figure.dpi=96",
]

# change index.rst to contents.rst for custom landing page feature
root_doc = "contents"

html_additional_pages = {
    "index": "index.html",
}


def setup(app):
    app.add_css_file("css/custom.css")  # may also be an URL
    # Register a sphinx.ext.autodoc.between listener to ignore everything between lines that contain the word COMMENT
    app.connect("autodoc-process-docstring", between("^.*COMMENT.*$", exclude=True))
    return app
