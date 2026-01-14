# Contributing to IRCA-Agent

Thank you for your interest in contributing to IRCA-Agent! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming and inclusive environment for all contributors.

## Getting Started

### Prerequisites

- Python 3.10-3.12
- [Poetry](https://python-poetry.org/) for dependency management
- Git

### Setup Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/irca-agent.git
cd irca-agent

# Install all dependencies including dev tools
poetry install --with dev

# Set up pre-commit hooks
pre-commit install

# Create a branch for your work
git checkout -b feature/your-feature-name
```

### Using DevContainer

For a consistent development environment, we recommend using the DevContainer:

1. Install [Docker](https://www.docker.com/) and [VS Code](https://code.visualstudio.com/)
2. Install the [DevContainers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
3. Open the project in VS Code
4. Click "Reopen in Container" when prompted

## Development Workflow

### Branch Naming

Use descriptive branch names:
- `feature/add-new-model-support` - New features
- `fix/training-memory-leak` - Bug fixes
- `docs/update-readme` - Documentation
- `refactor/simplify-trace-generator` - Code refactoring

### Making Changes

1. **Create a feature branch** from `main`
2. **Make small, focused commits** with clear messages
3. **Write tests** for new functionality
4. **Update documentation** as needed
5. **Run all checks** before submitting

### Running Checks

```bash
# Run tests
pytest

# Run linting
ruff check src/

# Run type checking
mypy src/

# Format code
ruff format src/

# Run all checks (recommended before committing)
pre-commit run --all-files
```

## Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications enforced by Ruff:

- Line length: 120 characters
- Use type hints for function signatures
- Use docstrings for public functions and classes

### Docstrings

Use Google-style docstrings:

```python
def generate_trace(
    available_functions: str,
    user_query: str,
    max_steps: int = 10,
) -> Trace:
    """
    Generate a complete agent trace.

    Args:
        available_functions: JSON string of available function definitions
        user_query: The user's query to respond to
        max_steps: Maximum number of function call iterations

    Returns:
        Trace containing all generated steps

    Raises:
        ValueError: If available_functions is not valid JSON
    """
    ...
```

### Type Hints

Always use type hints:

```python
def process_data(
    items: list[dict[str, Any]],
    filter_fn: Callable[[dict], bool] | None = None,
) -> list[dict[str, Any]]:
    ...
```

### Imports

Order imports as follows (enforced by Ruff):

```python
# Standard library
import json
import logging
from pathlib import Path

# Third-party
import torch
from pydantic import BaseModel

# Local
from core.domain import Step, Trace
from config import get_settings
```

## Testing

### Writing Tests

- Place tests in `tests/unit/` for unit tests
- Place tests in `tests/integration/` for integration tests
- Name test files `test_<module>.py`
- Name test functions `test_<description>`

Example:

```python
# tests/unit/test_trace.py
import pytest
from core.domain import Trace, ThoughtStep

class TestTrace:
    """Tests for Trace class."""

    def test_append_step(self):
        """Can append steps to a trace."""
        trace = Trace()
        step = ThoughtStep(thought="test", diff="Thought: test")
        trace.append(step)
        assert len(trace) == 1

    def test_empty_trace(self):
        """Empty trace has no steps."""
        trace = Trace()
        assert len(trace) == 0
```

### Using Fixtures

Share test data using fixtures in `conftest.py`:

```python
@pytest.fixture
def sample_functions():
    return '[{"name": "test_func", "description": "A test function"}]'

def test_with_functions(sample_functions):
    # Use the fixture
    assert "test_func" in sample_functions
```

### Test Coverage

Aim for at least 80% coverage on new code:

```bash
pytest --cov=src --cov-report=term-missing
```

## Submitting Changes

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(cli): add --verbose flag to generate command
fix(training): resolve memory leak during finetuning
docs(readme): update installation instructions
refactor(core): split trace_generator into focused modules
```

### Pull Request Process

1. **Update your branch** with the latest `main`
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Ensure all checks pass**
   ```bash
   pre-commit run --all-files
   pytest
   ```

3. **Create a pull request** with:
   - Clear title following conventional commits
   - Description of changes
   - Link to related issue(s)
   - Screenshots/examples if applicable

4. **Address review feedback** promptly

5. **Squash commits** if requested

## Issue Guidelines

### Reporting Bugs

Include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces
- Minimal reproducible example

### Requesting Features

Include:
- Clear description of the feature
- Use case / motivation
- Proposed solution (if any)
- Alternatives considered

### Questions

For questions, consider:
- Checking existing documentation
- Searching closed issues
- Opening a discussion (if available)

## 🙏 Thank You!

Every contribution, no matter how small, helps improve IRCA-Agent. Thank you for taking the time to contribute!
