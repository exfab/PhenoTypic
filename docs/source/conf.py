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
LIGHT_LOGO_PATH = './_static/assets/400x150/gradient_logo_exfab.svg'
DARK_LOGO_PATH = './_static/assets/400x150/gradient_logo_exfab.svg'

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
    'class_members',
    "sphinx_togglebutton"
]

autosummary_generate = True
# autosummary_imported_members = True

autodoc_default_options = {
    'members'          : True,  # Document class members
    'undoc-members'    : True,  # Include undocumented members
    'private-members'  : False,  # Include private members (e.g., `_method`)
    'show-inheritance' : True,  # Show class inheritance in docs
    'inherited-members': True,  # Include inherited members
    # 'member-order'     : 'bysource',
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


# Auto-generate downloadables documentation
def generate_downloadables_rst(app):
    import ast
    import os
    import re

    # Get directories relative to conf.py
    source_dir = os.path.abspath(os.path.dirname(__file__))
    downloadables_dir = os.path.join(source_dir, '_downloadables')
    output_file = os.path.join(source_dir, 'downloadables.rst')

    # Check if directory exists
    if not os.path.exists(downloadables_dir):
        print(f"Warning: {downloadables_dir} does not exist. Skipping downloadables generation.")
        return

    content = []
    content.append("Downloadable Scripts")
    content.append("====================")
    content.append("")
    content.append("This page contains downloadable scripts and utilities for PhenoTypic.")
    content.append("")
    content.append(".. grid:: 1 1 2 2")
    content.append("    :gutter: 3")
    content.append("")

    def extract_bash_description(filepath):
        """Extract title and description from bash script comments."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None, None

        title = None
        description_lines = []

        # Look for comment blocks (skip shebang and empty lines)
        in_comment_block = False
        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip shebang and empty lines at start
            if i == 0 and stripped.startswith('#!'):
                continue
            if not stripped:
                continue

            # Check if it's a comment line
            if stripped.startswith('#'):
                # Skip separator lines (===, ---, etc.)
                comment_content = stripped[1:].strip()
                if not comment_content or re.match(r'^[=\-]+$', comment_content):
                    continue

                # Extract title from first meaningful comment
                if title is None and comment_content:
                    # Check if it looks like a title (short, no period, capitalized)
                    if len(comment_content) < 100 and not comment_content.endswith('.'):
                        title = comment_content
                    else:
                        description_lines.append(comment_content)
                else:
                    # Collect description lines
                    if comment_content:
                        description_lines.append(comment_content)
                        # Stop at first empty line after collecting some content
                        if len(description_lines) >= 2:
                            # Check if next non-empty line is not a comment
                            for j in range(i + 1, min(i + 3, len(lines))):
                                next_stripped = lines[j].strip()
                                if next_stripped and not next_stripped.startswith('#'):
                                    break
                            else:
                                continue
                            break

        description = ' '.join(description_lines) if description_lines else None
        return title, description

    for filename in sorted(os.listdir(downloadables_dir)):
        if not (filename.endswith('.py') or filename.endswith('.sh')):
            continue

        filepath = os.path.join(downloadables_dir, filename)

        title = filename
        description = "No description available."

        if filename.endswith('.py'):
            # Handle Python files
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    docstring = ast.get_docstring(tree)
            except Exception as e:
                print(f"Error parsing {filename}: {e}")
                docstring = None

            if docstring:
                lines = docstring.strip().split('\n')
                # Try to extract a title from the first line or ReST header
                first_line = lines[0].strip()

                # Check for Title underline style
                # Title
                # =====
                if len(lines) > 1 and len(lines[1].strip()) >= len(first_line) and set(lines[1].strip()) == {'='}:
                    title = first_line
                    # Description starts after the header
                    desc_lines_raw = lines[2:]
                else:
                    # Just use filename as title if no clear header
                    # Or use the first line if it looks like a title
                    desc_lines_raw = lines

                # Extract description (first paragraph)
                desc_lines = []
                started = False
                for line in desc_lines_raw:
                    stripped = line.strip()
                    if not started:
                        if stripped:
                            started = True
                            desc_lines.append(stripped)
                    else:
                        if not stripped:
                            # Empty line indicates end of paragraph
                            break
                        desc_lines.append(stripped)

                if desc_lines:
                    description = ' '.join(desc_lines)

        elif filename.endswith('.sh'):
            # Handle bash files
            bash_title, bash_description = extract_bash_description(filepath)
            if bash_title:
                title = bash_title
            if bash_description:
                description = bash_description

        # Add card
        content.append(f"    .. grid-item-card:: {title}")
        content.append(f"        :shadow: md")
        content.append("")
        content.append(f"        {description}")
        content.append("")
        content.append("        +++")
        content.append(f"        :download:`Download script <_downloadables/{filename}>`")
        content.append("")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))
    print(f"Generated {output_file}")


def setup(app):
    app.connect('builder-inited', generate_downloadables_rst)
