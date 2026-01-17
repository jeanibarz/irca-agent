---
name: run-tests
description: Run pytest tests with various options - unit tests, integration tests, specific files, patterns, or full suite. Use when the user wants to run tests, verify code changes, or check for regressions.
allowed-tools: Bash, Read, Glob, Grep
---

# Run Tests

Execute the project test suite using pytest with various configurations.

## Test Structure

- **Unit tests**: `tests/unit/` (~390 tests) - Fast, isolated tests
- **Integration tests**: `tests/integration/` (~130 tests) - CLI commands, pipelines, component interactions
- **Total**: 520+ tests

## Commands

### Run All Tests
```bash
poetry run pytest
```

### Run Unit Tests Only (Fast Feedback)
```bash
poetry run pytest tests/unit/ -v
```

### Run Integration Tests Only
```bash
poetry run pytest tests/integration/ -v
```

### Run Specific Test File
```bash
poetry run pytest tests/unit/test_<name>.py -v
```

### Run Tests by Name Pattern
```bash
# Run all tests containing "diversity" in their name
poetry run pytest -k diversity -v

# Multiple patterns
poetry run pytest -k "settings or config" -v
```

### Run Specific Test Class or Method
```bash
# Specific class
poetry run pytest tests/unit/test_steps.py::TestTrace -v

# Specific method
poetry run pytest tests/unit/test_steps.py::TestTrace::test_trace_to_string -v
```

### Stop on First Failure (Debugging)
```bash
poetry run pytest -x -v
```

### Quick Smoke Test (No Coverage)
```bash
poetry run pytest tests/unit/ -x --no-cov -q
```

## Default Behavior

The `pyproject.toml` configures pytest with:
- Coverage for `src/` directory (automatic)
- HTML coverage report in `htmlcov/`
- Branch coverage enabled
- Asyncio mode enabled

## Usage Examples

When user says:
- "Run the tests" -> `poetry run pytest`
- "Run unit tests" -> `poetry run pytest tests/unit/`
- "Test the diversity module" -> `poetry run pytest -k diversity -v`
- "Quick test check" -> `poetry run pytest tests/unit/ -x --no-cov -q`

## Important Notes

1. **Coverage is automatic** - No need to add `--cov` flags
2. **GPU tests may be slow** - Integration tests involving models take longer
3. **Fixtures in conftest.py** - Common fixtures available for mocking
