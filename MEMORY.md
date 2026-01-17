
# Project Session Memory

**Last Updated:** 2026-01-17
**Last Commit:** `fix(core): ensure prompt is stripped from generation output` - Fixed prompt echoing.
**Branch:** cleanup/remove-irrelevant-files
**Task ID:** chat-template-formatting

---

## Current Task

**Goal:** Run augmentation impact experiment to determine if data augmentation improves model performance.
**Status:** ✅ Complete
**Experiment:** exp001-augmentation-impact
**Methodology:** ADR-008 - Same number of training steps, compare train/test perplexity
**Task Type:** experiment/evaluation

### Experiment State

| Item | Status | Details |
|------|--------|---------|
| Dataset split | ✅ Done | 152 train, 38 test (seed=42) |
| train_formatted | ✅ Done | 152 samples, no augmentation |
| train_augmented | ✅ Done | 436 samples, with augmentation |
| test_formatted | ✅ Done | 38 samples, English only |
| test_augmented | ✅ Done | 111 samples, multilingual |
| **ADR-008 CLI Tools** | ✅ Done | All phases implemented |
| Baseline model | ✅ Done | Mistral 7B v0.3, 3 epochs (114 steps) |
| Augmented model | ✅ Done | Mistral 7B v0.3, 1 epoch (109 steps) |
| Evaluation | ✅ Done | Both test sets evaluated |

### Experiment Results (exp001-augmentation-impact)

#### English Test Set (38 samples)

| Metric | Baseline | Augmented | Delta |
|--------|----------|-----------|-------|
| Best Checkpoint | checkpoint-76 | checkpoint-76 | - |
| Test Perplexity | 5.76 | 6.15 | **+6.7% (worse)** |
| Generalization Gap | 44.0% | 51.2% | +7.2% |

#### Augmented Test Set (111 samples, multilingual)

| Metric | Baseline | Augmented | Delta |
|--------|----------|-----------|-------|
| Best Checkpoint | checkpoint-38 | checkpoint-76 | - |
| Test Perplexity | 5.47 | 5.48 | **~0% (same)** |

**Key Finding:** Augmentation does not improve performance on either test set. On English-only test, augmented model is 6.7% worse. On multilingual test, both perform similarly.

### What I Did

1.  **ADR-008 Implementation (2026-01-17)**:
    *   **Phase 1: CLI Extensions** - Added `--output`, `--batch-size`, `--lora-r`, `--save-steps`, `--seed` to `irca finetune run`.
    *   **Phase 2: Experiment Commands** - Created `irca experiment eval-checkpoints` and `irca experiment compare`.
    *   **Phase 3: Bootstrap CI** - Created `src/diversity/statistics.py` with `bootstrap_confidence_interval()`.
    *   **Phase 4: Report Generation** - Added `generate_experiment_comparison_report()` to `src/diversity/report.py`.
    *   All unit tests passing (19 tests in `test_statistics.py`).

2.  **Experimental Framework for Augmentation (2026-01-17)**:
    *   Created ADR-007: Experimental Framework for Augmentation Impact Analysis.
    *   Created ADR-008: Fair Comparison Methodology for Augmentation Experiments.
    *   Implemented `irca dataset split` command in `src/cli/commands/dataset.py`.
    *   Split dataset: 152 train, 38 test (80/20, seed=42).
    *   Created 3 formatted datasets: train_formatted, train_augmented, test_formatted.
    *   Created experiment configuration and README in `experiments/exp001-augmentation-impact/`.

2.  **Chat Template Consistency (2026-01-17 - Phase 2)**:
    *   Added `convert_irca_to_chat_template()` to `src/diversity/utils.py` for evaluation.
    *   Added `apply_chat_template=True` parameter to all perplexity/evaluation functions.
    *   Updated `irca dataset diversity` CLI with `--no-chat-template` flag.
    *   **Removed** `format_variation` augmentation step (broke chat template parsing - see ADR-006).
    *   Cleaned up all config files (`configs/*.json`) to remove disabled format_variation entries.
    *   Updated all tests to use valid augmentation steps only.

2.  **Chat Template Implementation (2026-01-17)**:
    *   Created `src/formatting/chat_template.py` with utilities for model-specific formatting.
    *   Updated `src/finetuning/model_finetuning.py` to use `formatting_func` with chat templates.
    *   Updated `src/server/routers/generation.py` to use `format_for_inference()`.
    *   Updated `src/server/routers/synthetic.py` to use model-specific stop sequences.
    *   Key functions: `irca_to_messages()`, `apply_chat_template()`, `format_for_inference()`, `get_stop_sequences()`.

2.  **Frontend Polish**:
    *   Migrated to Tailwind CSS v4.
    *   Added **Available Tools** to Sidebar.
    *   Implemented **Conversation History** list in Sidebar.
    *   Implemented **"New Chat"** functionality.
    *   **Output Sanitization**: Automatically strips `<|wait|>` and other control tokens from the display.

2.  **Synthetic Red-Teaming**:
    *   Implemented `POST /v1/synthetic/query` endpoint with "Feasible" and "Infeasible" modes.
    *   **Fixed Generation**: Updated `ModelManager` to slice output tokens strictly (removing prompt echo).

3.  **Robust Backend**:
    *   **Async/Locking**: Prevented race conditions in model loading.
    *   **Persistence**: Created `src/server/routers/conversations.py` to save/load chats from `data/sessions/`.
    *   **Ejection**: Allowed manual GPU memory clearing.

4.  **Documentation**:
    *   Updated `docs/REQUIREMENTS.md` with Playgroung requirements including `FR-PLAY-07` (Dual Model support).
    *   Updated `docs/TRACEABILITY_MATRIX.md`.

### Key Findings

- **GPU Memory Constraint (CRITICAL)**: NEVER run multiple GPU-intensive tasks concurrently. The system cannot handle multiple models loaded in GPU memory at the same time. Always run model training, evaluation, and inference tasks sequentially, waiting for one to complete before starting another.
- **Augmentation Impact**: Based on exp001 results, translation augmentation does not improve test perplexity and may hurt performance on English-only test sets. Consider task-specific augmentation strategies.
- **Multiprocessing + CUDA Trap**: Avoid initializing CUDA (e.g., loading `transformers` model) before calling `dataset.map(num_proc=X)`. Forked child processes can deadlock or crash if they inherit a process state that has touched the GPU. Always perform diversification/preprocessing before loading the LLM.
- **Improved Dataset Parsing**: Standard agent traces contain multiple tool calls. Parsing must be robust to multiple `<|wait|>` tokens. Using specific section markers (e.g., `EXAMPLE:`, `### FUNCTIONS AVAILABLE`) is safer than blindly splitting on control tokens.
- **Sequential Model Memory**: Static (Offline) diversification using CPU models is highly efficient if performed *before* loading the main LLM. This "Unload-Between-Stages" pattern prevents RAM/VRAM exhaustion on mid-tier hardware.
- **Qwen3 Integration**: Verified compatibility with Qwen3 models following environment upgrades to `transformers 5.0.0rc` and `bitsandbytes 0.49.1`.
- **Cross-Lingual Training**: To train robust agents, augmenting the dataset with translated queries/answers while keeping reasoning traces in English is highly effective. Offline diversification ensures clean GPU state for training.

### Files Created/Modified

| File | Purpose |
|------|---------|
| `src/cli/commands/experiment.py` | **NEW**: Experiment evaluation commands (eval-checkpoints, compare). |
| `src/diversity/statistics.py` | **NEW**: Bootstrap CI and statistical utilities. |
| `src/cli/commands/finetune.py` | Extended with --output, --batch-size, --lora-r, --save-steps, --seed. |
| `src/diversity/report.py` | Added `generate_experiment_comparison_report()`. |
| `src/cli/__init__.py` | Registered experiment command group. |
| `tests/unit/test_statistics.py` | **NEW**: Unit tests for statistics module. |
| `src/formatting/chat_template.py` | **NEW**: Model-agnostic to chat template conversion utilities. |
| `src/finetuning/model_finetuning.py` | Updated to use `formatting_func` with chat templates at training time. |
| `src/server/routers/generation.py` | Updated to use `format_for_inference()` for inference. |
| `src/server/routers/synthetic.py` | Updated to use model-specific stop sequences. |
| `src/server/model_manager.py` | Progress capture & StopOnSequences with prompt length tracking. |
| `src/dataset_generation/translator.py` | `TranslationService` using Opus-MT models. |
| `src/config/settings.py` | Added `AugmentationConfig` fields for multilingual support. |
| `ui/src/components/Sidebar.tsx` | Reverted to single model UI (removed tabs). |

---

## Session Progress

### Completed This Session
- ✅ Configured separate Frontend (Vite) and Backend (FastAPI).
- ✅ Implemented Tool visualization and Synthetic Red-Teaming.
- ✅ Implemented Robust Backend (Async, Eject, Lock).
- ✅ Implemented Persistent Conversation History.
- ✅ Implemented Output Token Sanitization.
- ✅ Fixed Generation Output (Prompt Echoing).
- ✅ **New**: Added Unit Tests for Function Augmentation (FR-GEN-03).
- ✅ **New**: Reverted to Single Model Architecture - Simplified Sidebar and Routers.
- ✅ **New**: Implemented Reference Highlighting (FR-PLAY-08) - Interactive tool citations in Final Answer.
- ✅ **New**: Implemented Model Loading Progress Bar (RFC-002) - Granular Tqdm capture + SSE.
- ✅ **New**: Implemented Chat Template Formatting for training and inference (model-agnostic data storage).
- ✅ **New**: Created `src/formatting/chat_template.py` with `irca_to_messages()`, `apply_chat_template()`, `format_for_inference()`.
- ✅ **New**: Updated finetuning to use `formatting_func` that applies model-native chat templates at training time.
- ✅ **New**: Updated inference endpoint to use `format_for_inference()` for consistent prompt formatting.
- ✅ **New**: Added model-specific stop sequences via `get_stop_sequences()`.
- ✅ **New**: Created ADR-006: Chat Template Consistency Across Pipeline.
- ✅ **New**: Phase 2 Implementation - Added chat template conversion to diversity evaluation.
- ✅ **New**: Removed `format_variation` augmentation step (breaks parsing after chat template conversion).
- ✅ **New**: Updated COMPLETION_MARKERS with chat template-specific markers (Qwen, Llama3, Mistral).
- ✅ **New**: Cleaned up all config files - removed disabled format_variation entries.
- ✅ **New**: Updated all unit/integration tests to use valid augmentation steps only.
- ✅ **New**: Self-reflection completed - documented learnings in EXPERIENCES.md.
- ✅ **New**: Created ADR-007 (Experimental Framework) and ADR-008 (Fair Comparison Methodology).
- ✅ **New**: Implemented `irca dataset split` command for reproducible train/test splits.
- ✅ **New**: Created experiment exp001-augmentation-impact with datasets prepared.
- ✅ **New**: Implemented ADR-008 CLI tools (all 4 phases complete).
- ✅ **New**: Extended `irca finetune run` with `--output`, `--batch-size`, `--lora-r`, `--save-steps`, `--seed`.
- ✅ **New**: Created `irca experiment eval-checkpoints` for checkpoint evaluation.
- ✅ **New**: Created `irca experiment compare` for baseline vs augmented comparison.
- ✅ **New**: Created `src/diversity/statistics.py` with bootstrap confidence intervals.
- ✅ **New**: Added experiment comparison report generation to `src/diversity/report.py`.
- ⏳ **Pending**: Train baseline and augmented models on Mistral 7B v0.3.
- ⏳ **Pending**: Evaluate models and compare perplexity.
- **Previous**: Optimized diversification to use **1 GPU worker** - reduced total wait to ~2 mins.
- ✅ **New**: Fixed `parse_corrected_agent_trace` to handle multi-step traces and preserve English reasoning steps.
- ✅ **New**: Reordered CLI workflow to diversify dataset *before* loading LLM (Resource Optimization/CUDA fix).
- ✅ **New**: Upgraded `transformers` (v5.0.0rc) and `bitsandbytes` (v0.49.1) for Qwen3 support.
- ✅ **New**: Set default LoRA $r=64$ and Gradient Checkpointing to ensure 8B model fits on 24GB GPUs.
- ✅ **New**: Implemented auto-validation logging in diversification to detect parsing failures early.
- ✅ **New**: Updated `docs/REQUIREMENTS.md` and `docs/FINETUNING.md` to reflect Offline Diversification strategy.
- ✅ **New**: Cleaned up all orphaned CPU worker processes.

### Task Validation

-   **Command**: `npm run dev` + `python -m src.server.main`.
-   **Result**:
    -   Synthetic Queries now appear cleanly (without the prompt text).
    -   Chat history survives page reloads.
    -   "New Chat" clears context correctly.
    -   **Augmentation Tests**: `pytest tests/unit/test_augmentation.py` passes with coverage.

---

## Next Steps

1.  **Train Baseline Model (Mistral 7B v0.3)**:
    ```bash
    poetry run irca finetune run \
      -m mistral-v3 \
      -d experiments/exp001-augmentation-impact/train_formatted \
      -o experiments/exp001-augmentation-impact/models/baseline \
      --epochs 3 --batch-size 4 --lora-r 64 --save-steps 38 --seed 42
    ```

2.  **Train Augmented Model (Mistral 7B v0.3)**:
    ```bash
    poetry run irca finetune run \
      -m mistral-v3 \
      -d experiments/exp001-augmentation-impact/train_augmented \
      -o experiments/exp001-augmentation-impact/models/augmented \
      --epochs 1 --batch-size 4 --lora-r 64 --save-steps 38 --seed 42
    ```

3.  **Evaluate All Checkpoints**:
    ```bash
    # Baseline checkpoints
    poetry run irca experiment eval-checkpoints \
      -c experiments/exp001-augmentation-impact/models/baseline \
      --train-set experiments/exp001-augmentation-impact/train_formatted \
      --test-set experiments/exp001-augmentation-impact/test_formatted \
      -o experiments/exp001-augmentation-impact/results/baseline.json

    # Augmented checkpoints
    poetry run irca experiment eval-checkpoints \
      -c experiments/exp001-augmentation-impact/models/augmented \
      --train-set experiments/exp001-augmentation-impact/train_augmented \
      --test-set experiments/exp001-augmentation-impact/test_formatted \
      -o experiments/exp001-augmentation-impact/results/augmented.json
    ```

4.  **Compare Results**:
    ```bash
    poetry run irca experiment compare \
      -b experiments/exp001-augmentation-impact/results/baseline.json \
      -a experiments/exp001-augmentation-impact/results/augmented.json \
      -o experiments/exp001-augmentation-impact/results/comparison.html \
      --format html
    ```

**Reference:**
- `experiments/exp001-augmentation-impact/README.md`
- `docs/adrs/ADR-008-fair-augmentation-comparison-methodology.md`
- `PLAN.md` (implementation details)
2.  **Test Chat Template Formatting**:
    -   Re-finetune Qwen3-4B model with new chat template formatting.
    -   Verify model stops generation properly at inference.
3.  **RFC Phase 2: Function Execution Sandbox**:
    -   Current "Tools" are dummy definitions.
    -   Goal: Actually execute python code (e.g. `get_stock_price`) in a safe sandbox (Docker/gVisor).
4.  **LLM-as-a-Judge**:
    -   Automate the "synthetic query -> generation -> verify" loop using a stronger model.

---

## Files to Read After Memory Reset

1.  `MEMORY.md`
2.  `EXPERIENCES.md`
3.  `GEMINI.md`
