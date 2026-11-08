.PHONY: install dev test test-fast test-cov lint format typecheck benchmark benchmark-smoke benchmark-check benchmark-full clean build docs example verify all help

# Default Python interpreter
PYTHON ?= python3

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in production mode
	$(PYTHON) -m pip install .

dev:  ## Install package in development mode with dev dependencies
	$(PYTHON) -m pip install -e ".[dev]"

test:  ## Run all tests
	$(PYTHON) -m pytest tests/ -v

test-fast:  ## Run tests excluding slow tests
	$(PYTHON) -m pytest tests/ -v -m "not slow"

test-cov:  ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ -v --cov=zgq --cov-report=html --cov-report=term

lint:  ## Run linting checks
	$(PYTHON) -m flake8 zgq/ tests/ --max-line-length=100
	$(PYTHON) -m isort zgq/ tests/ --check-only --diff

format:  ## Format code with black and isort
	$(PYTHON) -m black zgq/ tests/ benchmarks/ examples/
	$(PYTHON) -m isort zgq/ tests/ benchmarks/ examples/

typecheck:  ## Run type checking with mypy
	$(PYTHON) -m mypy zgq/ --ignore-missing-imports

benchmark:  ## Run canonical benchmark at 10K into results/latest
	$(PYTHON) -m benchmarks.run_benchmarks --scale 10k --out-dir results/latest

benchmark-full:  ## Run canonical benchmark at 10K and 100K into results/latest
	$(PYTHON) -m benchmarks.run_benchmarks --scale 10k 100k --out-dir results/latest

benchmark-smoke:  ## Tiny-scale benchmark run plus schema check
	rm -rf .smoke_results
	$(PYTHON) -m benchmarks.run_benchmarks --scale tiny --n-queries 50 --repeats 2 \
		--ef 32 64 --nprobe 4 16 --zgq-m 16 24 --out-dir .smoke_results
	$(PYTHON) -m benchmarks.check_results .smoke_results/summary.json
	rm -rf .smoke_results

benchmark-check:  ## Validate committed canonical results against the schema
	@for f in results/*/summary.json; do $(PYTHON) -m benchmarks.check_results $$f --canonical || exit 1; done

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.nbc" -delete
	find . -type f -name "*.nbi" -delete

build:  ## Build package for distribution
	$(PYTHON) -m build

docs:  ## Generate documentation
	@echo "Documentation is in docs/ directory"
	@echo "View API.md and THEORY.md for documentation"

example:  ## Run quickstart example
	$(PYTHON) examples/quickstart.py

verify:  ## Verify installation and run quick tests (includes benchmark-smoke)
	$(PYTHON) -c "from zgq import ZGQIndex, ZGQConfig; print('ZGQ v8 imported successfully')"
	$(PYTHON) -c "import numpy as np; from zgq import ZGQIndex, ZGQConfig; \
		config = ZGQConfig(); \
		index = ZGQIndex(config); \
		data = np.random.randn(500, 64).astype(np.float32); \
		index.build(data); \
		q = np.random.randn(64).astype(np.float32); \
		idx, dist = index.search(q, k=5); \
		print(f'Search test passed: found {len(idx)} neighbors')"
	$(MAKE) benchmark-smoke

all: format lint typecheck test  ## Run all checks (format, lint, typecheck, test)
