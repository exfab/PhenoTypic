# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Signals to phenotypic's plotly init that nbsphinx-executed kernels should
# emit HTML-renderable Plotly output instead of the default JSON MIME bundle.
# See src/phenotypic/_core/.../_accessor_dash_handler.py.
export PHENOTYPIC_DOCS_BUILD = 1

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

apidoc:
	sphinx-apidoc -o source/api_reference ../src/phenotypic --module-first --separate --no-toc

html: apidoc
	rm -f source/api_reference/phenotypic.rst
	make clean
	sphinx-build -n -b html source build/html
