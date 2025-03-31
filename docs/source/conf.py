# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import sphinx_autosummary_accessors
sys.path.insert(0, os.path.abspath('../../src'))

project = 'PhenoScope'
copyright = '2025, ExFAB BioFoundry'
author = 'Alexander Nguyen'

# Variables
github_url = 'https://github.com/Wheeldon-Lab/PhenoScope#'
LIGHT_LOGO_PATH = './_static/assets/500x225/no_background_svg/light_logo_sponsor.svg'
DARK_LOGO_PATH = './_static/assets/500x225/no_background_svg/dark_logo_sponsor.svg'

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
    'sphinx.ext.autosectionlabel',
    'nbsphinx',
    'sphinx_gallery.gen_gallery',
    'sphinx_autosummary_accessors',
    'sphinx_design',
    'myst_nb'
]

autosummary_generate = True

autodoc_default_options = {
    'members': True,  # Document class members
    'undoc-members': True,  # Include undocumented members
    'private-members': False,  # Include private members (e.g., `_method`)
    'show-inheritance': True,  # Show class inheritance in docs
    'inherited-members': True,  # Include inherited members
}

autodoc_typehints = 'both'
autodoc_member_order = 'groupwise'

templates_path = ['_templates', sphinx_autosummary_accessors.templates_path]

# nbsphinx configuration
nbsphinx_execute = 'auto'
nbsphinx_allow_errors = False
nbsphinx_kernel_name = 'python3'

# myst_nb configuration
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_nb_output_stderr = "remove"

# sphinx-gallery configuration
sphinx_gallery_conf = {
    'examples_dirs': '../../examples',   # path to your example scripts
    'gallery_dirs': 'auto_examples',     # path to where to save gallery generated output
    'filename_pattern': '/example_',     # pattern to match example files
    'ignore_pattern': '__init__\.py',   # pattern to ignore
    'plot_gallery': 'True',             # generate plots
    'thumbnail_size': (400, 300),       # thumbnail size
    'download_all_examples': True,      # download all examples as a zip file
    'line_numbers': True,               # show line numbers in code blocks
    'remove_config_comments': True,     # remove config comments from code blocks
    'capture_repr': ('_repr_html_', '__repr__'),  # capture representations for display
}
exclude_patterns = ['_build', '**.ipynb_checkpoints', '*.ipynb', 'auto_examples']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Themes: 'sphinxawesome_theme', 'furo', 'pydata_sphinx_theme'
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

if html_theme == 'pydata_sphinx_theme':
    html_logo = LIGHT_LOGO_PATH
    html_theme_options = {
        "logo": {
            "alt_text": "PhenoScope",
            "link": "index",
            "image_light":LIGHT_LOGO_PATH,
            "image_dark":DARK_LOGO_PATH
        },
        "icon_links": [
            {
                "name": "GitHub",
                "url": github_url,
                "icon": "fa-brands fa-github",
            }
        ],
        "use_edit_page_button": False,
        "show_toc_level": 3
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

