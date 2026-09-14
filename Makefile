VENV ?= .venv
# Every target runs from the venv once `make venv` has created it, and from the
# system interpreter before that.
PYTHON ?= $(shell test -x $(VENV)/bin/python && echo $(VENV)/bin/python || echo python3)
PORT ?= 8000
DATA_DIR ?= data/sample
NODE ?= node

.DEFAULT_GOAL := help

.PHONY: help venv dev dev-api dev-web install install-vit install-dev check test test-all test-vit web web-test run samples evaluate clean

help:
	@echo "venv         create .venv and install the API and test dependencies into it"
	@echo "dev          start the API and the page locally and print their URLs"
	@echo "dev-api      start only the API (page served at /)"
	@echo "dev-web      start only the standalone page server"
	@echo "install      install the API and test dependencies"
	@echo "install-vit  add torch, torchvision and transformers"
	@echo "install-dev  add pytest and httpx"
	@echo "check        byte-compile every module, then run the tests"
	@echo "test         run the Python test suite"
	@echo "web-test     run the page's node test suite"
	@echo "test-all     run both suites"
	@echo "test-vit     run the checkpoint tests (needs weights)"
	@echo "web          build the page into frontend/dist"
	@echo "run          serve the API and the page on PORT (default $(PORT))"
	@echo "samples      write placeholder images to $(DATA_DIR)"
	@echo "evaluate     score $(DATA_DIR) and print the metrics"
	@echo "clean        drop caches, generated reports and the page build"

venv:
	python3 -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -r requirements-dev.txt

dev:
	bash scripts/dev.sh

dev-api:
	bash scripts/dev.sh --api-only

dev-web:
	bash scripts/dev.sh --web-only

install:
	$(PYTHON) -m pip install -r requirements.txt

install-vit:
	$(PYTHON) -m pip install -r requirements-vit.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

check:
	$(PYTHON) -m compileall -q app scripts tests
	$(PYTHON) -m pytest

test:
	$(PYTHON) -m pytest

web-test:
	$(NODE) --test "frontend/tests/**/*.test.mjs"

test-all: test web-test

test-vit:
	$(PYTHON) -m pytest -m vit

web:
	$(NODE) frontend/scripts/build.mjs

run:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port $(PORT)

samples:
	$(PYTHON) -m scripts.make_samples --output $(DATA_DIR)

evaluate:
	$(PYTHON) -m scripts.evaluate_model --data-dir $(DATA_DIR)

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache reports frontend/dist
