# Test Strategy

## Levels of Testing

### 1. Unit Testing (Level 1)
**Scope**: Individual functions and independent classes.
- **Location**: `tests/unit/`
- **Focus**:
  - **Utils**: Verify string manipulation, JSON parsers, and shufflers (`src/core/utils.py`).
  - **Prompt Building**: Verify that prompts are constructed correctly from templates (`src/core/prompt_builder.py`).
  - **Domain Objects**: Verify integrity of `Trace`, `Step` objects.
  - **Step Generators**: Mock the `guidance` model object to verify that generators construct the correct guidance programs.

### 2. Integration Testing (Level 2)
**Scope**: Interaction between subsystems (TraceGenerator + Domain + Mock Model).
- **Location**: `tests/integration/`
- **Focus**:
  - **Trace Logic**: Instantiate a `TraceGenerator` with a dummy/mock model. Verify that it correctly transitions states (Thought -> Action -> Output).
  - **Mocking Strategy**: Since running a real LLM is slow/expensive in CI, creating a `MockGuidanceModel` that returns deterministic strings for specific prompts is crucial.

### 3. System / Smoke Testing (Level 3)
**Scope**: End-to-end execution of CLI commands.
- **Location**: `tests/e2e/` (or manual smoke tests)
- **Focus**:
  - **CLI Execution**: Use `click.testing.CliRunner` to invoke `irca generate` and `irca finetune`.
  - **TinyLlama Run**: Perform a real generation run using `tinyllama` (small model) on a very small dataset (1-2 samples) to ensure the full pipeline (loading -> generating -> saving) works without crashing.

## Test Data Management

- **Fixtures**: Use `tests/conftest.py` to provide:
  - Sample Function Schemas (JSON).
  - Sample User Queries.
  - Mock Guidance Models.
- **Golden Files**: Store expected prompt outputs in `tests/data/` to prevent regression in prompt formatting.

## Recommended Test Cases to Add

1.  **`test_trace_generator_flow`**:
    - *Scenario*: Simulate a "success" path where model calls function X and then answers.
    - *Input*: Mock model programmed to output "call function X".
    - *Assertion*: Verify trace contains `FunctionCallStep` and `FunctionOutputStep`.

2.  **`test_negative_example_generation`**:
    - *Scenario*: Run generation with `generate_trace_missing_function`.
    - *Input*: Mock model.
    - *Assertion*: Ensure the target function is effectively removed from the prompt passed to the model.

3.  **`test_cli_argument_parsing`**:
    - *Scenario*: Run `irca generate --invalid-arg`.
    - *Assertion*: Ensure graceful exit and help message.

## Continuous Integration (CI)

- **Trigger**: On Pull Request.
- **Jobs**:
  1. `lint`: `ruff check`
  2. `type-check`: `mypy`
  3. `unit-tests`: `pytest tests/unit`
  4. `build`: Verify package installs (`poetry install`).

---

**Navigation:**
- [Traceability Matrix](TRACEABILITY_MATRIX.md)
- [Functional Requirements](../requirements/FUNCTIONAL.md)
- [Non-Functional Requirements](../requirements/NON_FUNCTIONAL.md)
