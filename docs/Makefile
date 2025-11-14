# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# cat command on top fixes relative locations in the README file
%: Makefile
	# remove old copies
	rm -f README-copy.md CODEOWNERS-copy.md CONTRIBUTING-copy.md CODE_OF_CONDUCT-copy.md
	# create new copies
	cp ../README.md README-copy.md
	cp ../CODEOWNERS.md CODEOWNERS-copy.md
	cp ../CONTRIBUTING.md CONTRIBUTING-copy.md
	cp ../CODE_OF_CONDUCT.md CODE_OF_CONDUCT-copy.md
	sed -i.bak -e 's|docs/_static/|_static/|g' -e 's|/docs/_static/|_static/|g' README-copy.md
	sed -i.bak 's|<img src=_static/logo_firecamp.png width="200">|<img src="_static/logo_firecamp.png" width="200" class="readme-logo">|g' README-copy.md
	rm -f README-copy.md.bak
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
