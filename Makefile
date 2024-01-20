.PHONY: all clean install base-install install-conda install-poetry


######################################################################
# Important, this block allows us to pass arguments to the makefile, and
# allows missing arguments to be replaced with a default value.
%:
	@:

args = `arg="$(filter-out $@,$(MAKECMDGOALS))" && echo $${arg:-${1}}`
######################################################################

include .env

export $(shell sed 's/=.*//' .env)


all: clean install

clean:
	@find . -name '*.pyc' -exec rm -rf {} \;
	@find . -name '__pycache__' -exec rm -rf {} \;
	@find . -name 'Thumbs.db' -exec rm -rf {} \;
	@find . -name '*~' -exec rm -rf {} \;
	rm -rf .cache
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf htmlcov
	rm -rf .tox/
	rm -rf docs/_build

install: base-install install-conda install-poetry

base-install:
	@echo "Installing base dependencies"
	@pip install --upgrade pip

install-conda:
	@echo "Installing conda dependencies"
	@conda env remove --name ml-problems
	@conda env create -f environment.yml

install-poetry:
	@echo "Installing poetry dependencies"
	@pip install poetry
	@poetry env use 3.7
	@poetry lock --no-update
	@poetry install