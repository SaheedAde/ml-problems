.PHONY: all clean install base-install install-conda update update-conda run-examples


######################################################################
# Important, this block allows us to pass arguments to the makefile, and
# allows missing arguments to be replaced with a default value.
%:
	@:

args = `arg="$(filter-out $@,$(MAKECMDGOALS))" && echo $${arg:-${1}}`
######################################################################

include .env

export $(shell sed 's/=.*//' .env)


all: clean install install base-install install-conda update update-conda run-examples

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
	@conda env remove --name ml-problems

install: base-install install-conda

base-install:
	@echo "Installing base dependencies"
	@pip install --upgrade pip

install-conda:
	@echo "Installing conda dependencies"
	@conda env create -f environment.yml

update: update-conda

update-conda:
	@echo "Updating conda dependencies"
	@conda env update -f environment.yml

run-examples:
	@echo "Running the project on examples"
	@conda run -n ml-problems python examples_main.py