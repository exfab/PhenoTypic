# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'PhenoScope'
copyright = '2025, ExFAB BioFoundry'
author = 'Alexander Nguyen'

# Try to get the version from phenoscope, but use a default if not available
try:
    import phenoscope
    version = str(phenoscope.__version__)
except ImportError:
    version = '0.1.0'  # Default version if phenoscope is not installed
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.ifconfig',
    'sphinx.ext.autosectionlabel'
]

autodoc_default_options = {
    'members': True,  # Document class members
    'undoc-members': True,  # Include undocumented members
    'private-members': False,  # Include private members (e.g., `_method`)
    'show-inheritance': True,  # Show class inheritance in docs
    'inherited-members': True,  # Include inherited members
}

autodoc_typehints = 'both'
autodoc_member_order = 'groupwise'

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Themes: 'sphinxawesome_theme', 'furo', 'pydata_sphinx_theme'
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
# html_css_files = [
#     'custom.css',
# ]

if html_theme == "furo":
    html_theme_options = {
        'light_css_variables': {
        },
        'dark_css_variables': {
        }
    }

if html_theme == 'sphinxawesome_theme':
    html_theme_options = {
        "logo_light":"../assets/logo_background_svg/PhenoScopeLogo.svg",
        "logo_dark":"../assets/logo_background_svg/PhenoScopeLogo-DarkMode.svg"
    }

if html_theme == 'pydata_sphinx_theme':
    html_theme_options = {
    }

# Napoleon Settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

