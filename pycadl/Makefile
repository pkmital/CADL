.PHONY: help clean clean-docs clean-tox clean-pyc clean-build list test test-all coverage docs release sdist

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-cov - remove Coverage file artifacts"
	@echo "clean-docs - remove Docs file artifacts"
	@echo "clean-tox - remove Tox artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "test-all - run tests on every Python version with tox"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "release - package and upload a release to devpi"
	@echo "sdist - package"

clean: clean-build clean-cov clean-docs clean-tox clean-pyc

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info

clean-cov:
	rm -rf htmlcov/

clean-docs:
	$(MAKE) -C docs clean

clean-tox:
	rm -rf .tox

clean-pyc:
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

lint:
	tox -e flakes

test:
	tox

test-all:
	tox
	tox -e docs
	tox -e flakes
	tox -e cov

coverage: clean-cov
	tox -e cov
	open htmlcov/index.html

docs: clean-docs
	$(MAKE) -C docs html
	open docs/_build/html/index.html

release: clean
	devpi upload

sdist: clean
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist
