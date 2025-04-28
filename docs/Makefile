#===============================================================================
# Copyright 2014 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = "-j`nproc`"
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

create_rst_examples:
	@echo "Creating rST files for oneAPI Examples..."
	python3 rst_examples.py
	@echo "Done"

doxygen:
	cd doxygen/oneapi && doxygen

parse-doxygen: doxygen
	mkdir -p build
	python -m dalapi.doxypy.cli doxygen/oneapi/doxygen/xml --compact > build/tree.yaml

.PHONY: help Makefile create_rst_examples doxygen parse-doxygen

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
html: create_rst_examples Makefile
	@$(SPHINXBUILD) -W --keep-going -w docbuild-log.txt -n -t use_intelname -b $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html-github: create_rst_examples Makefile
	@$(SPHINXBUILD) -W --keep-going -w docbuild-log.txt -n -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html-test: create_rst_examples Makefile
	@$(SPHINXBUILD) -W --keep-going -w docbuild-log.txt -n -b $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
