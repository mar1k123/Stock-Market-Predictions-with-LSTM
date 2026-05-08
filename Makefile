#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = stock-market-lstm
PYTHON_VERSION = 3.11
UV = uv
UV_INSTALL_URL = https://astral.sh/uv/install.sh

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install project dependencies (including dev tools) via uv
.PHONY: requirements
requirements: ensure_uv
	$(UV) sync --all-extras

## Ensure uv is installed (installs with curl when missing)
.PHONY: ensure_uv
ensure_uv:
	@command -v $(UV) >/dev/null 2>&1 || (echo "uv not found, installing..." && curl -LsSf $(UV_INSTALL_URL) | sh)
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint: ensure_uv
	$(UV) run ruff format --check
	$(UV) run ruff check

## Format source code with ruff
.PHONY: format
format: ensure_uv
	$(UV) run ruff check --fix
	$(UV) run ruff format



## Run tests
.PHONY: test
test: ensure_uv
	$(UV) run python -m pytest tests


## Create project virtual environment via uv
.PHONY: create_environment
create_environment: ensure_uv
	$(UV) venv --python $(PYTHON_VERSION)
	@echo ">>> Virtual environment created in .venv"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(UV) run python stock_market_lstm/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
