# Development Guide

This guide covers local development workflows for `irca-agent`. For contribution guidelines, please see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Environment Setup

### 1. Dependency Management
We use **Poetry** for dependency management.

```bash
# Install core dependencies
poetry install

# Install dev dependencies (pytest, ruff, black)
poetry install --with dev

# Activate virtualenv
poetry shell
```

### 2. DevContainers
The project includes a `.devcontainer` configuration. This is the recommended way to ensure a consistent environment tailored for Python + Cuda development.
- Open the project in VS Code.
- Reopen in Container.

## Common Workflows

### Running Tests
Tests are located in `tests/`.

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_trace.py

# Run with coverage
pytest --cov=src
```

### Linting & Formatting
We use `ruff` (replacing black, isort, flake8).

```bash
# Check for issues
ruff check src/

# Auto-fix issues
ruff check --fix src/

# Format code
ruff format src/
```

### Debugging Generation
When debugging the prompt or generation logic:

1. Use the verbose flag `-v` with the CLI:
   ```bash
   irca -v generate traces --limit 1
   ```
   This will print detailed logs of the generation steps.

2. Inspect `datasets/` output. The generated JSON files contain the full `prompt` field, which allows you to see exactly what was sent to the model.

## Adding New Features

### Adding a New Step Type
1. Update `src/core/domain/steps.py` to include the new class (e.g., `ReflectionStep`).
2. Update `StepType` enum.
3. specific generator function in `src/core/generation/step_generators.py`.
4. Update `TraceGenerator` logic to invoke the new step.

### Adding a New Model Family
1. Check `src/config/settings.py` -> `get_model_config`.
2. Add the HuggingFace ID and chat template mapping.
3. Ensure `guidance` supports the model architecture.

## Release Process
1. Update version in `pyproject.toml`.
2. Update `CHANGELOG.md` (if exists).
3. Tag the commit and push.
