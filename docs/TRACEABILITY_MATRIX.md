# Traceability Matrix

This matrix maps Functional Requirements (FR) to their Implementation components and verifying Test cases.

| Req ID | Requirement Description | Implementation Component(s) | Verifying Test(s) |
|--------|--------------------------|----------------------------|-------------------|
| **FR-GEN-01** | Trace Generation Loop | `src/core/generation/trace_generator.py`<br>`src/core/domain/trace.py` | `tests/unit/test_trace_generator.py`*<br>`tests/integration/test_full_loop.py`* |
| **FR-GEN-02** | Guidance Constraints | `src/core/generation/step_generators.py` | `tests/unit/test_step_generators.py`* |
| **FR-GEN-03** | Function Removal | `src/core/generation/trace_generator.py` (augment logic)<br>`src/core/utils.py` | `tests/unit/test_function_augmentation.py`* |
| **FR-GEN-04** | Function Shuffling | `src/core/utils.py` -> `shuffle_json_functions` | `tests/unit/test_utils.py` |
| **FR-GEN-05** | Prompt Randomization | `src/core/prompt_builder.py` | `tests/unit/test_prompt_builder.py` |
| **FR-CLI-01** | Command Interface | `src/cli/main.py`<br>`src/cli/commands/*` | Manual Verification / CLI Smoke Tests |
| **FR-CLI-02** | Verbose Mode | `src/cli/utils.py` (logging setup) | Manual Verification |
| **FR-FT-01** | Q-LoRA Training | `src/finetuning/model_finetuning.py` | `tests/integration/test_finetuning.py`* |
| **FR-FT-02** | Model Presets | `src/config/settings.py` -> `get_model_config` | `tests/unit/test_settings.py`* |
| **FR-FT-03** | Configurable Params | `src/config/settings.py` | `tests/unit/test_settings.py`* |
| **FR-DATA-01** | HF Integration | `src/dataset_generation/hf_utils.py`*<br>`src/cli/commands/dataset.py` | Manual Integration Test |

*> Symbol denotes components/tests that are logically inferred to exist or should exist based on the architecture analysis. Files marked with `*` in the Test column indicate recommended coverage gaps if not already present.

## Gap Analysis

Based on the current analysis of `tests/`:

1.  **Unit Tests**: There is good coverage for utilities (`test_utils.py`) and prompt building (`test_prompt_builder.py`).
2.  **Missing coverage**:
    *   Explicit tests for `trace_generator.py` core logic (mocking the LLM) are critical for **FR-GEN-01**.
    *   Tests for `finetuning` logic are likely missing or only exist as manual scripts.
    *   CLI command tests (using `click.testing.CliRunner`) would ensure **FR-CLI-01** reliability.

## Traceability to Files

- **Logic**: `src/core/generation/` satisfies GEN requirements.
- **Config**: `src/config/settings.py` satisfies FT-02, FT-03 and NFR-CODE-02.
- **CLI**: `src/cli/commands/` satisfies CLI requirements and DATA-01.
