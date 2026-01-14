# Tests

Automated tests for IRCA-Agent.

## Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests
│   ├── test_steps.py        # Step models and Trace
│   ├── test_utils.py        # Utility functions
│   ├── test_prompt_builder.py  # Prompt building
│   ├── test_settings.py     # Configuration
│   └── test_constants.py    # Generation constants
└── integration/             # Integration tests (future)
```

## Running Tests

```bash
# Install dev dependencies
poetry install --with dev

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_steps.py

# Run specific test class
pytest tests/unit/test_steps.py::TestTrace

# Run specific test
pytest tests/unit/test_steps.py::TestTrace::test_trace_to_string

# Run with verbose output
pytest -v

# Run and stop on first failure
pytest -x
```

## Writing Tests

### Naming Conventions

- Test files: `test_<module>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<description>`

### Using Fixtures

Common fixtures are available in `conftest.py`:

```python
def test_with_sample_functions(sample_functions_json):
    """Use sample function definitions."""
    functions = json.loads(sample_functions_json)
    assert len(functions) == 3

def test_with_mock_settings(mock_settings):
    """Use mocked Settings object."""
    assert mock_settings.lora_r == 128
```

### Test Categories

1. **Unit Tests** (`tests/unit/`)
   - Test individual functions and classes in isolation
   - Use mocks for external dependencies
   - Should be fast and focused

2. **Integration Tests** (`tests/integration/`)
   - Test interactions between components
   - May require GPU for model tests
   - Mark with `@pytest.mark.slow` if needed

## Coverage Goals

| Module | Target | Current |
|--------|--------|---------|
| `core.domain` | 90% | TBD |
| `core.utils` | 90% | TBD |
| `core.prompt_builder` | 80% | TBD |
| `config.settings` | 80% | TBD |
| Overall | 50% | TBD |

## Markers

```python
# Skip slow tests
@pytest.mark.slow
def test_full_training():
    ...

# Skip tests requiring GPU
@pytest.mark.gpu
def test_model_inference():
    ...
```

Run with markers:
```bash
# Skip slow tests
pytest -m "not slow"

# Only GPU tests
pytest -m gpu
```