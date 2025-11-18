# c2s-mini Test Suite

This directory contains comprehensive tests for the c2s-mini package.

## Test Files

- `conftest.py` - Pytest configuration with slow test markers
- `test_utils.py` - Tests for core transformation functions in utils.py
- `test_data.py` - Tests for C2SData class
- `test_prompts.py` - Tests for prompt formatting functions
- `test_model.py` - Integration tests for C2SModel (marked as slow)
- `test_tasks.py` - Tests for high-level API functions (marked as slow)

## Running Tests

### Run All Fast Tests (Excluding Model Tests)
```bash
pytest -m "not slow" -v
```

### Run All Tests (Including Slow Model Tests)
```bash
pytest -v
```

### Run Specific Test File
```bash
pytest tests/test_utils.py -v
```

### Run Tests with Coverage
```bash
pytest --cov=c2s_mini --cov-report=html
```

## Test Categories

### Fast Tests (~few seconds)
- `test_utils.py` - Data transformation logic
- `test_data.py` - Data wrapper functionality
- `test_prompts.py` - Prompt formatting

### Slow Tests (~minutes, requires model download)
- `test_model.py` - Model loading and inference
- `test_tasks.py` - End-to-end API tests

## Requirements

All tests require the package to be installed with dev dependencies:
```bash
pip install -e ".[dev]"
```

Slow tests additionally require downloading the 160M parameter model from HuggingFace, which will happen automatically on first run.

## Fixtures

### small_adata
- Provides a small test dataset (10 cells × 50 genes)
- Used in utils and data tests
- Based on PBMC3K dataset from scanpy

### small_csdata
- Provides a C2SData object (5 cells)
- Used in tasks tests
