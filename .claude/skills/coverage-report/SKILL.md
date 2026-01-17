---
name: coverage-report
description: Generate and analyze test coverage reports. Use when the user wants to see code coverage, identify untested code, check coverage percentages, or view coverage reports.
allowed-tools: Bash, Read, Glob, Grep
---

# Coverage Report

Generate and analyze test coverage for the codebase.

## Quick Commands

### Generate Coverage Report (Default)
Coverage is automatically generated when running pytest:
```bash
poetry run pytest
```

### Coverage with Missing Lines Detail
```bash
poetry run pytest --cov-report=term-missing
```

### Generate HTML Report Only
```bash
poetry run pytest --cov-report=html --cov-report=
```

### View HTML Coverage Report
```bash
xdg-open htmlcov/index.html
```

### Check Coverage Threshold
```bash
# Fail if coverage below 80%
poetry run pytest --cov-fail-under=80

# Fail if coverage below specific percentage
poetry run pytest --cov-fail-under=70
```

## Output Locations

| Type | Location | Description |
|------|----------|-------------|
| Terminal | stdout | Summary with percentages |
| HTML | `htmlcov/index.html` | Interactive line-by-line visualization |
| JSON | `htmlcov/status.json` | Machine-readable data |

## Coverage Configuration

From `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["src"]
branch = true
omit = ["tests/*", "*/__pycache__/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if __name__ == .__main__.:"
]
```

## Interpreting Results

### Terminal Output Example
```
Name                          Stmts   Miss Branch BrPart  Cover
---------------------------------------------------------------
src/cli/__init__.py              15      2      4      1    85%
src/cli/commands/dataset.py     120     10     40      5    89%
...
---------------------------------------------------------------
TOTAL                          2500    250    800     80    88%
```

### HTML Report Features
- **Green**: Covered lines
- **Red**: Uncovered lines
- **Yellow**: Partially covered branches
- Click on files to see line-by-line coverage

## Usage Examples

When user says:
- "Show me coverage" -> `poetry run pytest --cov-report=term-missing`
- "What's not covered?" -> Run tests then `xdg-open htmlcov/index.html`
- "Check we have 80% coverage" -> `poetry run pytest --cov-fail-under=80`
- "Generate coverage report" -> `poetry run pytest` (automatic)

## Analyzing Coverage Gaps

To find untested code:

1. **Run tests with missing lines**:
   ```bash
   poetry run pytest --cov-report=term-missing
   ```

2. **Check specific module**:
   ```bash
   poetry run pytest tests/unit/test_<module>.py --cov=src/<module> --cov-report=term-missing
   ```

3. **Open HTML for detailed view**:
   ```bash
   xdg-open htmlcov/index.html
   ```

## CI/CD Integration

For pipelines, use threshold enforcement:
```bash
poetry run pytest --cov-fail-under=80 --cov-report=xml
```

This fails the build if coverage drops below 80% and generates XML for CI tools.
