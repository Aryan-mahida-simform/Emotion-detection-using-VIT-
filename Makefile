PYTHON ?= python3
PORT ?= 8000
DATA_DIR ?= data/sample
NODE ?= node

.DEFAULT_GOAL := help

.PHONY: help install install-vit install-dev check test test-all test-vit web web-test run samples evaluate clean

help:
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
