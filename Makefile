sinclude .env

all: help

install-linters: ## install the linters
	pip install black==23.10.0 ruff==0.6.9 mypy==1.9.0 types-psutil

install: install-linters

mypy:
	python3 -m mypy --check-untyped-defs --follow-imports=silent --exclude third_party/ train.py

lint:
	python3 -m ruff format --check --exclude fastgen/third_party/
	python3 -m ruff check --exclude fastgen/third_party/

format:
	python3 -m ruff format --exclude fastgen/third_party/
	python3 -m ruff check --fix --exclude fastgen/third_party/

install-fastgen:
	python3 -m pip install -e .

pytest:
	ulimit -n 4096 && python3 -m pytest --ignore=FASTGEN_OUTPUT --ignore=runs --ignore=tmp --ignore third_party

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
