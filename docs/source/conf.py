# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

""" Color Palette
Primary (Sky Blue)
    #00AEEF
    Bright sky blue for branding focus
Accent 1
    #0077B6
    Deeper blue for headers/navs
Accent 2
    #90E0EF
    Soft sky tint for backgrounds
Background
    #F4FAFD
    Very light blue-white background
Text (Dark)
    #023047
    Almost black with blue undertone
Text (Light)
    #FFFFFF
    For light-on-dark components
Link
    #219EBC
    Soft blue for hyperlinks

"""

import os
import sys
import sphinx_autosummary_accessors

sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('./_extensions'))

project = 'PhenoTypic'
copyright = '2025, ExFAB BioFoundry'
author = 'Alexander Nguyen'

# Variables
github_url = 'https://github.com/Wheeldon-Lab/PhenoScope#'
LIGHT_LOGO_PATH = './_static/assets/200x150/light_logo_sponsor.svg'
DARK_LOGO_PATH = './_static/assets/200x150/dark_logo_sponsor.svg'

# Try to get the version from PhenoTypic, but use a default if not available
try:
    import phenotypic

    version = str(phenotypic.__version__)
except ImportError:
    version = '0.1.0'  # Default version if PhenoTypic is not installed
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
    'sphinx_autosummary_accessors',
    'sphinx_design',
    'myst_nb',
    'class_members',
    "sphinx_togglebutton"
]

autosummary_generate = True

autodoc_default_options = {
    'members'          : True,  # Document class members
    'undoc-members'    : True,  # Include undocumented members
    'private-members'  : False,  # Include private members (e.g., `_method`)
    'show-inheritance' : True,  # Show class inheritance in docs
    'inherited-members': True,  # Include inherited members
}

autodoc_typehints = 'both'
autodoc_typehints_format = 'short'
autodoc_member_order = 'groupwise'

templates_path = ['_templates', sphinx_autosummary_accessors.templates_path]

# Suppress specific warnings
suppress_warnings = [
    'toc.not_readable',  # Suppress warnings about documents not in toctree
    'autosectionlabel.*',  # Suppress duplicate label warnings
    'autodoc.duplicate_object',  # Suppress duplicate object warnings
]

# Exclude patterns - don't process these files/directories
exclude_patterns = ['_build', '**.ipynb_checkpoints', '**/auto_examples']

# nbsphinx configuration
nbsphinx_execute = 'auto'
nbsphinx_allow_errors = True
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

# Disable strict HTML5 assertion for broken references
html5_writer = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Themes: 'sphinxawesome_theme', 'furo', 'pydata_sphinx_theme'
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

if html_theme == 'pydata_sphinx_theme':
    html_title = "PhenoTypic"
    html_theme_options = {
        "subtitle": "A modular framework for bioimage analysis and visualization"
    }
    html_logo = LIGHT_LOGO_PATH
    html_theme_options = {
        "logo"                : {
            "alt_text"   : "PhenoTypic",
            "link"       : "index",
            "image_light": LIGHT_LOGO_PATH,
            "image_dark" : DARK_LOGO_PATH
        },
        "icon_links"          : [
            {
                "name": "GitHub",
                "url" : github_url,
                "icon": "fa-brands fa-github",
            }
        ],
        "use_edit_page_button": False,
        "show_toc_level"      : 3,

        "navigation_with_keys": True,
        "show_prev_next"      : False,
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

# Type aliases for cleaner documentation
python_type_aliases = {
    'matplotlib.axes._axes.Axes': 'matplotlib.axes.Axes',
    'matplotlib.figure.Figure'  : 'matplotlib.figure.Figure'
}

intersphinx_mapping = {
    "python"    : ("https://docs.python.org/3", None),
    "numpy"     : ("https://numpy.org/doc/stable/", None),
    "pandas"    : ("https://pandas.pydata.org/docs/", None),
    "scipy"     : ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn"   : ("https://scikit-learn.org/stable/", None),
    "skimage"   : ("https://scikit-image.org/docs/stable/", None),
    "h5py"      : ("https://docs.h5py.org/en/stable/", None),
    "plotly"    : ("https://plotly.com/python-api-reference/", None),
    "colour"    : ("https://colour.readthedocs.io/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}
