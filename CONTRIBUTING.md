# Contributing to c2s-mini

Thank you for your interest in contributing to c2s-mini! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Prioritize the community's best interests

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of single-cell RNA sequencing
- Familiarity with PyTorch and transformers

### Finding Issues to Work On

- Check the [issue tracker](https://github.com/salzcamino/c2s-mini/issues) for open issues
- Look for issues labeled `good first issue` for beginner-friendly tasks
- Issues labeled `help wanted` are great for contributors
- Feel free to propose new features or improvements

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/c2s-mini.git
cd c2s-mini

# Add upstream remote
git remote add upstream https://github.com/salzcamino/c2s-mini.git
```

### 2. Create a Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n c2s-mini python=3.9
conda activate c2s-mini
```

### 3. Install Development Dependencies

```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# Or install from requirements
pip install -r requirements.txt
pip install pytest jupyter ipykernel
```

### 4. Verify Installation

```bash
# Test import
python -c "import c2s_mini; print(c2s_mini.__version__)"

# Run fast tests
pytest tests/ -m "not slow" -v
```

## Project Structure

```
c2s-mini/
├── c2s_mini/           # Main package
│   ├── __init__.py     # Package initialization
│   ├── utils.py        # Core transformation utilities
│   ├── data.py         # C2SData wrapper class
│   ├── model.py        # C2SModel wrapper class
│   ├── prompts.py      # Prompt formatting functions
│   └── tasks.py        # High-level task functions
├── tests/              # Test suite
│   ├── conftest.py     # Pytest configuration
│   ├── test_utils.py   # Tests for utils.py
│   ├── test_data.py    # Tests for data.py
│   ├── test_model.py   # Tests for model.py
│   ├── test_prompts.py # Tests for prompts.py
│   └── test_tasks.py   # Tests for tasks.py
├── examples/           # Example notebooks
│   ├── basic_usage.ipynb
│   └── cell_type_prediction.ipynb
├── docs/               # Documentation
├── CLAUDE.md           # Implementation guide
├── README.md           # Main documentation
└── pyproject.toml      # Package configuration
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Prefer single quotes for strings, double quotes for docstrings
- **Imports**: Group in order: stdlib, third-party, local

### Type Hints

- Use type hints for all function signatures
- Prefer modern syntax: `list[str]` over `List[str]` (Python 3.9+)
- For Python 3.8 compatibility, use `from __future__ import annotations`

```python
from __future__ import annotations

def process_cells(
    adata: AnnData,
    n_genes: int = 200
) -> list[str]:
    """Process cells and return sentences."""
    pass
```

### Docstring Format

Use NumPy-style docstrings:

```python
def example_function(param1: str, param2: int = 10) -> list[str]:
    """
    Brief description of function.

    More detailed description if needed. Can span multiple lines
    and include equations, references, etc.

    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)

    Returns:
        List of processed strings

    Raises:
        ValueError: When param1 is empty

    Example:
        >>> result = example_function("test", 5)
        >>> print(len(result))
        5
    """
    pass
```

### Code Quality Tools

We recommend using these tools (optional but encouraged):

```bash
# Install code quality tools
pip install black isort flake8 mypy

# Format code
black c2s_mini/ tests/
isort c2s_mini/ tests/

# Check style
flake8 c2s_mini/ tests/

# Type checking
mypy c2s_mini/
```

## Testing Guidelines

### Writing Tests

- **Location**: Place tests in `tests/test_<module>.py`
- **Naming**: Test functions should start with `test_`
- **Coverage**: Aim for >80% code coverage for new features
- **Types of tests**:
  - **Unit tests**: Test individual functions
  - **Integration tests**: Test module interactions
  - **Edge cases**: Test boundary conditions and error handling

### Test Structure

```python
import pytest
from c2s_mini import C2SData

def test_feature_happy_path():
    """Test normal operation."""
    # Setup
    data = create_test_data()

    # Execute
    result = process_data(data)

    # Assert
    assert result is not None
    assert len(result) > 0

def test_feature_edge_case():
    """Test edge case."""
    # Test with empty input
    result = process_data([])
    assert result == []

def test_feature_error_handling():
    """Test error conditions."""
    with pytest.raises(ValueError, match="Invalid input"):
        process_data(None)
```

### Running Tests

```bash
# Run all tests (including slow model tests)
pytest tests/ -v

# Run only fast tests (skip model loading)
pytest tests/ -m "not slow" -v

# Run specific test file
pytest tests/test_utils.py -v

# Run with coverage report
pytest tests/ --cov=c2s_mini --cov-report=html

# Run tests matching a pattern
pytest tests/ -k "test_vocabulary" -v
```

### Test Markers

- `@pytest.mark.slow`: Marks tests that take >5 seconds (model loading)
- Use for integration tests involving model inference

```python
@pytest.mark.slow
def test_model_generation():
    """Test model generation (slow)."""
    model = C2SModel(device='cpu')
    result = model.generate_from_prompt("test")
    assert isinstance(result, str)
```

## Documentation

### Documentation Requirements

All contributions should include:

1. **Docstrings**: All public functions, classes, and methods
2. **Type hints**: All function signatures
3. **Examples**: At least one example in docstrings for public APIs
4. **README updates**: If adding new features
5. **CHANGELOG**: Entry describing the change

### Example Notebooks

If adding new features, consider adding example notebooks:

```bash
# Create new notebook
jupyter notebook examples/my_new_feature.ipynb
```

Guidelines for notebooks:
- Include clear markdown explanations
- Show realistic use cases
- Include output cells
- Keep execution time reasonable (<2 minutes)

### Building Documentation

If adding to formal docs:

```bash
# Future: Sphinx documentation
cd docs
make html
```

## Submitting Changes

### Branch Naming

Use descriptive branch names:

- `feature/add-cell-clustering`
- `fix/memory-leak-in-embedding`
- `docs/improve-api-reference`
- `test/add-edge-cases`

### Commit Messages

Follow conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Examples:**

```
feat(model): add support for batch size auto-tuning

Implement automatic batch size selection based on available GPU memory.
This improves performance on systems with varying GPU capacities.

Closes #42
```

```
fix(utils): handle sparse matrices correctly in generate_vocabulary

Previously, dense conversion was happening unnecessarily for sparse matrices,
causing memory issues with large datasets.
```

### Pull Request Process

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/my-new-feature
   ```

3. **Make your changes**:
   - Write code
   - Add tests
   - Update documentation

4. **Run tests**:
   ```bash
   # Run all tests
   pytest tests/ -v

   # Check code style
   black --check c2s_mini/ tests/
   ```

5. **Commit changes**:
   ```bash
   git add .
   git commit -m "feat: add my new feature"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/my-new-feature
   ```

7. **Create Pull Request**:
   - Go to GitHub and create a PR
   - Fill out the PR template
   - Link related issues
   - Request review

### Pull Request Checklist

Before submitting, ensure:

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No merge conflicts with main
- [ ] Commit messages are clear
- [ ] PR description explains changes

## Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

Example: `0.1.0` → `0.2.0` (new feature) → `0.2.1` (bug fix)

### Release Checklist

For maintainers:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Create git tag: `git tag v0.2.0`
5. Push tag: `git push origin v0.2.0`
6. Create GitHub release
7. Build and upload to PyPI (future)

## Development Tips

### Performance Considerations

- Use sparse matrices for expression data
- Implement batch processing for model inference
- Profile code with `cProfile` for bottlenecks
- Use `torch.no_grad()` for inference

### Common Pitfalls

1. **Dense matrix conversion**: Always check if conversion is necessary
2. **Device mismatch**: Ensure tensors are on the correct device
3. **Memory leaks**: Clear CUDA cache with `torch.cuda.empty_cache()`
4. **Gene name casing**: Vocabulary uses uppercase gene names

### Debugging Tips

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check tensor device
print(f"Tensor device: {tensor.device}")

# Profile memory usage
import tracemalloc
tracemalloc.start()
# ... your code ...
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

## Getting Help

- **Questions**: Open a [discussion](https://github.com/salzcamino/c2s-mini/discussions)
- **Bugs**: Open an [issue](https://github.com/salzcamino/c2s-mini/issues)
- **Features**: Discuss in issues before implementing
- **Documentation**: Original Cell2Sentence [repository](https://github.com/vandijklab/cell2sentence)

## Recognition

Contributors will be:
- Listed in the repository
- Acknowledged in release notes
- Credited in publications (for significant contributions)

Thank you for contributing to c2s-mini!
