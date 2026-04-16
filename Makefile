SHELL := /usr/bin/bash

qc-env:
	pip install -r requirements.txt
	pip install -e src/quditkit-main_schmidt

nice:
	ruff check --fix --force-exclude src/ scripts/ tests/
	ruff format --force-exclude src/ scripts/ tests/
