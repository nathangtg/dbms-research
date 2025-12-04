.PHONY: install dev test lint format typecheck benchmark clean docs help

# Default Python interpreter
PYTHON ?= python3

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in production mode
	$(PYTHON) -m pip install .

dev:  ## Install package in development mode with all dependencies
	$(PYTHON) -m pip install -e ".[all]"

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

benchmark:  ## Run benchmark suite
	$(PYTHON) -m benchmarks.run_benchmarks --dataset 10k

benchmark-full:  ## Run full benchmark suite (all sizes)
	$(PYTHON) -m benchmarks.run_benchmarks --dataset 10k
	$(PYTHON) -m benchmarks.run_benchmarks --dataset 100k

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

publish-test:  ## Publish to TestPyPI
	$(PYTHON) -m twine upload --repository testpypi dist/*

publish:  ## Publish to PyPI
	$(PYTHON) -m twine upload dist/*

docs:  ## Generate documentation
	@echo "Documentation is in docs/ directory"
	@echo "View API.md and THEORY.md for documentation"

example:  ## Run quickstart example
	$(PYTHON) examples/quickstart.py

verify:  ## Verify installation and run quick tests
	$(PYTHON) -c "from zgq import ZGQIndex, ZGQConfig; print('ZGQ v8 imported successfully')"
	$(PYTHON) -c "import numpy as np; from zgq import ZGQIndex, ZGQConfig; \
		config = ZGQConfig(dimension=64); \
		index = ZGQIndex(config); \
		data = np.random.randn(100, 64).astype(np.float32); \
		index.build(data); \
		q = np.random.randn(64).astype(np.float32); \
		idx, dist = index.search(q, k=5); \
		print(f'Search test passed: found {len(idx)} neighbors')"

all: format lint typecheck test  ## Run all checks (format, lint, typecheck, test)
