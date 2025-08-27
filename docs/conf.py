import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

import mentat_lss
print(mentat_lss.__file__)
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mentat-lss'
copyright = '2025, Joe Adamo, Grace Gibbins, Annie Moore'
author = 'Joe Adamo, Grace Gibbins, Annie Moore'
root_doc = 'index'
release = '0.9.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", 
              "sphinx.ext.napoleon",
              "sphinx.ext.viewcode",
              'nbsphinx']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
