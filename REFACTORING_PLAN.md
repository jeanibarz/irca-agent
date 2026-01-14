# IRCA-Agent Refactoring Plan

**Created:** 2026-01-14
**Branch:** `cleanup/remove-irrelevant-files`

---

## Executive Summary

This document outlines a phased approach to refactoring the IRCA-Agent codebase. The goal is to modernize the project, improve maintainability, and prepare for future experiments with newer models and techniques.

---

## Current State Analysis

### Architecture Overview
```
irca-agent/
├── src/
│   ├── core/                      # Core logic
│   │   ├── trace_generator.py     # Main trace generation (429 lines) - COMPLEX
│   │   ├── prompt_builder.py      # Prompt construction (231 lines)
│   │   ├── step_factory.py        # Step types with Pydantic (83 lines) - CLEAN
│   │   ├── utils.py               # Utility functions (77 lines)
│   │   └── prompt/                # Prompt templates
│   ├── dataset_generation/        # Function definitions
│   │   ├── functions_factory.py   # Factory pattern (25 lines)
│   │   └── function_variants/     # GPT4 function schemas
│   ├── finetuning/
│   │   └── model_finetuning.py    # Training logic (236 lines)
│   └── defaults/
│       └── v1/training_args.py    # Training configuration
├── scripts/                       # Runnable scripts (some overlap)
│   ├── main.py                    # Dataset creation (legacy?)
│   ├── main_generator.py          # Main trace generation script
│   ├── create_irca_agent_dataset.py
│   ├── combine_datasets.py
│   ├── push_to_huggingface.py
│   └── config.py                  # Duplicate of training_args?
└── datasets/                      # Generated datasets
```

### Key Issues Identified

#### 1. **Dependency Management** 🔴 Critical
- `requirements.txt` and `pyproject.toml` are **out of sync**
- `requirements.txt` has pinned old versions (e.g., `trl==0.4.7`, `peft==0.8.2`)
- `pyproject.toml` has newer versions but missing key packages (`argilla`, `shortuuid`, `guidance`)
- No lockfile strategy (poetry.lock exists but not synced)

#### 2. **Hardcoded Paths** 🔴 Critical
- Multiple hardcoded `/workspace/` paths in:
  - `src/finetuning/model_finetuning.py` (lines 23-29)
  - `scripts/main_generator.py` (line 69)
  - `src/defaults/v1/training_args.py` (lines 37, 41)

#### 3. **Code Duplication** 🟡 High
- `scripts/config.py` duplicates `src/defaults/v1/training_args.py`
- Multiple Argilla initialization patterns across scripts
- Similar dataset loading patterns repeated in multiple scripts

#### 4. **Tight Coupling** 🟡 High
- `trace_generator.py` imports specific modules at module level
- Model loading is tightly coupled to `guidance` library
- No dependency injection or composition patterns

#### 5. **Missing Documentation** 🟡 High
- No root README.md
- No API documentation
- Limited inline documentation

#### 6. **No Tests** 🟡 High
- `tests/` directory exists but only contains README.md
- No unit tests, integration tests, or validation

#### 7. **Inconsistent Code Style** 🟢 Medium
- Mixed use of global variables (`iteration_nbr` in prompt_builder.py)
- Inconsistent logging (some use print, some use logger)
- Commented-out code blocks in many files

#### 8. **Legacy/Dead Code** 🟢 Medium
- Commented-out functions (e.g., `validate_info_completeness` in trace_generator.py)
- Multiple legacy scripts that seem unused

---

## Phased Refactoring Plan

### Phase 0: Pre-Refactoring Setup ✅ COMPLETE
**Goal:** Establish a baseline and development environment
**Time Estimate:** 1-2 hours
**Risk:** Low

- [x] Create backup
- [x] Create cleanup branch
- [x] Remove irrelevant files
- [x] Update .gitignore
- [x] **Commit current cleanup state**

```bash
git add -A
git commit -m "chore: cleanup irrelevant files and update .gitignore"
```

---

### Phase 1: Dependency Modernization ✅ COMPLETE
**Goal:** Unify and update dependencies
**Time Estimate:** 2-4 hours
**Risk:** Medium (may break compatibility)

#### 1.1 Consolidate to Poetry
- [x] Remove `requirements.txt` (keep `pyproject.toml` only)
- [x] Add missing dependencies to `pyproject.toml`:
  ```toml
  argilla = "^1.x"
  shortuuid = "^1.x"
  guidance = "^0.1.x"  # Check latest version
  jsonschema = "^4.x"
  pydantic = "^2.x"
  ```
- [ ] Update `pyproject.toml` with proper dev dependencies:
  ```toml
  [tool.poetry.group.dev.dependencies]
  pytest = "^7.x"
  ruff = "^0.1.x"
  mypy = "^1.x"
  ```

#### 1.2 Update Outdated Packages
- [x] Check for breaking changes in:
  - `trl` (0.4.7 → 0.13.0)
  - `transformers` (→ 4.47.0)
  - `argilla` (→ 2.8.0)
  - `guidance` (→ 0.2.0)
- [ ] Test finetuning pipeline after updates (deferred to validation)

#### 1.3 Update DevContainer
- [x] Update Dockerfile with Poetry
- [x] Update post-create script
- [x] Update devcontainer.json

---

### Phase 2: Configuration Refactoring ✅ COMPLETE
**Goal:** Centralize and externalize configuration
**Time Estimate:** 2-3 hours
**Risk:** Low

#### 2.1 Create Unified Config System
- [x] Create `src/config/` module:
  ```python
  # src/config/__init__.py
  from .settings import Settings

  # src/config/settings.py
  from pydantic_settings import BaseSettings
  from pathlib import Path

  class Settings(BaseSettings):
      workspace_dir: Path = Path("/workspace")
      models_dir: Path = Path("models")
      datasets_dir: Path = Path("datasets")

      # HuggingFace
      huggingface_token: str | None = None

      # Argilla
      argilla_api_url: str | None = None
      argilla_api_key: str | None = None

      # Training
      lora_r: int = 128
      lora_alpha: int = 64
      lora_dropout: float = 0.05
      num_train_epochs: int = 5
      learning_rate: float = 1e-3

      class Config:
          env_file = ".env"
          env_file_encoding = "utf-8"
  ```

#### 2.2 Remove Hardcoded Paths
- [x] Replace all `/workspace/` paths with config-based paths
- [x] Use `Path` objects instead of string concatenation
- [x] Support both local and container environments

#### 2.3 Consolidate Training Config
- [x] Remove duplicate `scripts/config.py`
- [x] Merge `src/defaults/v1/training_args.py` into `Settings`
- [x] Use model registry pattern for model configs

---

### Phase 3: Core Module Refactoring ✅ COMPLETE
**Goal:** Improve modularity and testability
**Time Estimate:** 4-6 hours
**Risk:** Medium

#### 3.1 Refactor `trace_generator.py` (429 lines → focused modules)
- [x] Split into focused modules:

```
src/core/
├── domain/              # Data models
│   ├── __init__.py
│   ├── steps.py         # Step types (from step_factory.py)
│   └── trace.py         # Trace model
├── generation/          # Trace generation
│   ├── __init__.py
│   ├── constants.py     # Prompt constants
│   ├── trace_generator.py   # Main orchestration
│   ├── step_generators.py   # Individual step generation
│   └── argilla.py       # Argilla integration
└── utils.py
```

#### 3.2 Introduce Dependency Injection
- [x] TraceGenerator accepts pre-initialized model
- [x] Protocol-based interface for models

```python
# Now supports dependency injection
class TraceGenerator:
    def __init__(self, model_name_or_path: str | None = None, model: LanguageModel | None = None):
        # Can inject model or load from path
```

#### 3.3 Remove Global State
- [x] Remove `iteration_nbr` global in `prompt_builder.py`
- [x] Use InstructionFormatter class for state management

#### 3.4 Add Type Hints
- [x] Add comprehensive type hints to all modules
- [x] Backwards compatibility maintained via re-exports

---

### Phase 4: Script Consolidation ✅ COMPLETE
**Goal:** Create a clean CLI interface
**Time Estimate:** 3-4 hours
**Risk:** Low

#### 4.1 Create CLI Entry Point
- [x] Created `src/cli/` module with Click-based CLI
- [x] Command groups: `generate`, `finetune`, `dataset`

```
irca --help
irca generate traces --model path/to/model --limit 10
irca finetune run --model-type mistral --epochs 5
irca dataset push --source dataset_name --target user/repo
```

#### 4.2 Update pyproject.toml
- [x] Entry point configured: `irca = "src.cli:cli"`
- [x] Added click dependency

#### 4.3 Deprecate Individual Scripts
- [x] Mark old scripts as deprecated in scripts/README.md
- [x] Add migration notes with CLI equivalents
- [ ] Eventually remove after transition period

---

### Phase 5: Testing Infrastructure ✅ COMPLETE
**Goal:** Add test coverage
**Time Estimate:** 4-6 hours
**Risk:** Low

#### 5.1 Setup Testing Framework
- [x] pytest configuration in pyproject.toml
- [x] conftest.py with shared fixtures
- [x] Unit and integration test directories

#### 5.2 Add Unit Tests
- [x] `test_steps.py` - Step models, Trace, create_step factory
- [x] `test_utils.py` - extract_and_remove, shuffle_json_functions
- [x] `test_prompt_builder.py` - build_full_prompt, format_instruction
- [x] `test_settings.py` - Settings, get_model_config, get_training_config
- [x] `test_constants.py` - Generation constants

#### 5.3 Add Integration Tests
- [ ] Test trace generation with mocked model (deferred)
- [ ] Test dataset loading/saving (deferred)
- [x] Test configuration loading

---

### Phase 6: Documentation ✅ COMPLETE
**Goal:** Create comprehensive documentation
**Time Estimate:** 2-3 hours
**Risk:** Low

#### 6.1 Create Root README.md
- [x] Complete rewrite with CLI documentation
- [x] Added project structure, features, and quick start guide

#### 6.2 API Documentation
- [x] Created `docs/api-reference.md`
- [x] Documented Config, Domain Models, TraceGenerator, and Utils
- [x] Updated `src/README.md` with module overview

#### 6.3 Developer Guide
- [x] Created `CONTRIBUTING.md`
- [x] Included development workflow, coding standards, and testing guide
- [x] Added `LICENSE` file

---

### Phase 7: Code Quality & CI/CD ✅ COMPLETE
**Goal:** Automate quality checks
**Time Estimate:** 2-3 hours
**Risk:** Low

#### 7.1 Add Linting & Formatting
- [x] Configured Ruff in `pyproject.toml`
- [x] Set strict MyPy rules for source code

#### 7.2 Add Pre-commit Hooks
- [x] Created `.pre-commit-config.yaml`
- [x] Added Ruff, MyPy, and standard hooks

#### 7.3 CI/CD
- [x] Created `.github/workflows/ci.yml`
- [x] Configured matrix testing for Python 3.10/3.11

---

### Phase 8: Validation ✅ COMPLETE
**Goal:** Verify full system functionality in clean environment
**Time Estimate:** 2 hours
**Risk:** Low

#### 8.1 Environment Setup
- [x] Create fresh virtual environment (Verified locally)
- [x] Install dependencies with Poetry

#### 8.2 Quality Checks
- [x] Run `pre-commit run --all-files`
- [x] Fix any linting/typing issues
- [x] Verify `irca --help` works

#### 8.3 Testing
- [x] Run full test suite
- [x] Verify coverage goals

---

## Recommended Execution Order

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3
   │                                   │
   │                                   ▼
   │           ◄─────────────────  Validation
   │                                   │
   │                                   ▼
   │           Phase 4 ◄─── Phase 5 ◄──┘
   │               │
   │               ▼
   └──────────► Phase 6 ──► Phase 7 ──► Done
```

### Validation Checkpoints

After each phase:
1. **Run existing scripts** - Ensure they still work
2. **Import check** - `python -c "from src.core import trace_generator"`
3. **Type check** - `mypy src/` (after Phase 3)
4. **Test suite** - `pytest` (after Phase 5)

---

## File Changes Summary

| File/Directory | Action | Phase |
|----------------|--------|-------|
| `requirements.txt` | Remove | 1 |
| `scripts/config.py` | Remove | 2 |
| `src/config/` | Create | 2 |
| `src/core/generation/` | Create | 3 |
| `src/core/models/` | Create | 3 |
| `src/cli/` | Create | 4 |
| `tests/test_*.py` | Create | 5 |
| `README.md` | Create | 6 |
| `.pre-commit-config.yaml` | Create | 7 |
| `.github/workflows/` | Create | 7 |

---

## Risk Mitigation

1. **Breaking Changes**: Always work in a branch, validate before merging
2. **Dependency Updates**: Pin versions in pyproject.toml, use lockfile
3. **Data Loss**: Keep datasets backed up, use HuggingFace Hub
4. **Model Compatibility**: Test with actual finetuning run before declaring success

---

## Success Criteria

- [ ] All existing functionality preserved
- [ ] Can run finetuning end-to-end on a test dataset
- [ ] Code passes linting and type checking
- [ ] At least 50% test coverage on core modules
- [ ] Documentation covers setup and usage
- [ ] No hardcoded paths remain
- [ ] Single source of truth for configuration

---

## Next Steps

1. **Review this plan** - Adjust phases as needed
2. **Commit current cleanup** - Create baseline
3. **Start Phase 1** - Dependency modernization
4. **Iterate** - Complete one phase at a time

Would you like me to start with any specific phase?
