# Contributing to ZGQ

Thank you for your interest in contributing to ZGQ (Zone-Guided Quantization)! This document provides guidelines for contributing to the project.

## Contact

- **Maintainer**: Nathan
- **Email**: nathan.n@imail.sunway.edu.my
- **Repository**: https://github.com/nathangtg/dbms-research/

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all backgrounds and experience levels.

## Getting Started

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/nathangtg/dbms-research.git
   cd dbms-research/v8
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .\.venv\Scripts\activate  # Windows
   ```

3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. Verify installation:
   ```bash
   pytest tests/
   ```

## Development Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `perf/description` - Performance improvements
- `refactor/description` - Code refactoring

### Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```

2. Make your changes

3. Run tests:
   ```bash
   pytest tests/
   ```

4. Format code:
   ```bash
   black zgq/ tests/
   isort zgq/ tests/
   ```

5. Check types:
   ```bash
   mypy zgq/
   ```

6. Commit your changes:
   ```bash
   git commit -m "feat: description of your changes"
   ```

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `perf:` - Performance improvement
- `refactor:` - Code refactoring
- `test:` - Adding tests
- `chore:` - Maintenance

## Code Style

### Python Style Guide

- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use isort for import sorting
- Add type hints to all functions

### Documentation

- Add docstrings to all public functions and classes
- Use NumPy-style docstrings
- Update README.md if adding new features

Example docstring:

```python
def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search for k nearest neighbors.
    
    Parameters
    ----------
    query : np.ndarray
        Query vector of shape (dimension,)
    k : int, default=10
        Number of neighbors to return
        
    Returns
    -------
    indices : np.ndarray
        Indices of k nearest neighbors
    distances : np.ndarray
        Distances to k nearest neighbors
        
    Examples
    --------
    >>> indices, distances = index.search(query, k=10)
    """
```

## Testing

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_zgq.py

# With coverage
pytest --cov=zgq tests/

# Skip slow tests
pytest -m "not slow" tests/
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use meaningful assertions with clear error messages

Example test:

```python
def test_index_build():
    """Test that index builds correctly."""
    from zgq import ZGQIndex, ZGQConfig
    
    config = ZGQConfig(M=16, ef_construction=200, use_pq=False, verbose=False)
    index = ZGQIndex(config)
    
    vectors = np.random.randn(1000, 128).astype(np.float32)
    index.build(vectors)
    
    assert index.n_vectors == 1000
    assert index.is_built
```

## Benchmarking

When making performance changes, run benchmarks:

```bash
python -m benchmarks.run_benchmarks --dataset 10k
```

Include benchmark results in your PR if performance is affected.

## Pull Requests

### Before Submitting

- [ ] All tests pass
- [ ] Code is formatted with Black
- [ ] Type hints are added
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated

### PR Description

Provide:
1. Summary of changes
2. Motivation/context
3. Test coverage
4. Benchmark results (if applicable)

### Review Process

1. PR will be reviewed by maintainers
2. Address feedback and update PR
3. Once approved, PR will be merged

## Reporting Issues

### Bug Reports

Include:
- ZGQ version
- Python version
- Operating system
- Minimal reproducible example
- Expected vs actual behavior

### Feature Requests

Include:
- Use case description
- Proposed API (if applicable)
- Alternative approaches considered

## Questions?

- Open a GitHub issue for questions
- Tag with `question` label

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
