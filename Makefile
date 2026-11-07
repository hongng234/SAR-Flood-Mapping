# Shortcuts for the main commands. Without `make`, run the python lines directly.
PY ?= .venv/Scripts/python.exe

.PHONY: setup download all test lint

setup:
	python -m venv .venv
	$(PY) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
	$(PY) -m pip install -r requirements.txt

download:
	$(PY) main.py download

all:
	$(PY) main.py all

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests main.py
