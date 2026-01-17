# Project Experience Log

> **READ THIS FIRST** - This file captures learnings across sessions.
>
> **When READING (session start):** Skim all sections to recall past learnings.
>
> **When UPDATING (after tasks):**
> 1. Identify what you learned (mistakes, patterns, codebase facts)
> 2. Add to the PERMANENT SECTION (Common Mistakes, Patterns That Work, etc.)
> 3. Do NOT just add summaries to "Recently Added" - that's a tracking section, not storage
> 4. Keep entries concise and specifically actionable.
>
> Keep under 500 lines. Remove deprecated/invalid entries when file grows.

**Last Updated:** 2026-01-17 (Chat Template Consistency Phase 2 Session)

---

## Environment Setup
<!-- Commands, paths, tools specific to this environment -->

- **Dependencies**: `sentencepiece` is required for Mistral v0.3 tokenizer.
- **Poetry**: Use `poetry run <command>` or `poetry shell`.
- **Workspace**: Root is `/home/jean/git/irca-agent`.

---

## Common Mistakes
<!-- Errors made and prevention strategies -->

| Mistake | Prevention | Added |
|---------|------------|-------|
| Passing `max_seq_length`/`packing` to `SFTTrainer.__init__` | Use `trl.SFTConfig` and pass these args there. New `trl` versions moved these from `__init__` to config. | 2026-01-15 |
| Missing `sentencepiece` for Mistral models | Add `sentencepiece` to `pyproject.toml` dependencies. | 2026-01-15 |
| `TypeError: SFTTrainer.__init__() got an unexpected keyword argument` | Check if the argument belongs in `SFTConfig` instead of `__init__`. | 2026-01-15 |
| Mistral v3 Finetuning Instability | High Learning Rate (1e-3) causes gradient explosion. Use `2e-4` or lower. Also ensure `tokenizer.padding_side='right'`. | 2026-01-15 |
| Mocking Local Imports inside Functions | Use the full source path `patch("module.submodule.ClassName")` instead of `patch("module_using_it.ClassName")` when the import is performed inside the function body. | 2026-01-15 |
| Model `.to(device)` in Tests | When mocking PyTorch-like models, ensure `.to()` returns `self` (the mock instance) to avoid breaking chain-calls to methods like `.generate()`. | 2026-01-15 |
| **The Mocking Trap** | Over-mocking complex internal logic (like prompt parsing/reconstruction) in tests can hide structural bugs. If the test mocks the very thing it should be verifying, it becomes a "self-fulfilling prophecy" that passes even when the real implementation is broken. | 2026-01-16 |
| **Mocking Local Imports** | When mocking a service or class, ensure you patch the correct namespace where it is *imported*, not just where it is defined. | 2026-01-16 |
| **Device-Specific Mocks** | Always mock `torch.cuda.is_available` and `.to(device)` calls when tests are intended to run in heterogeneous environments (CI vs. local). | 2026-01-16 |
| **Background Model Loading Conflicts** | Loading secondary models (like Opus-MT) on CUDA within `DataLoader` workers often leads to `RuntimeError`. Workers share GPU context, causing deadlocks. | 2026-01-17 |
| **CUDA + Multiprocessing (Fork) Trap** | Initializing CUDA (e.g. `AutoModel.from_pretrained`) BEFORE calling `dataset.map(num_proc=X)` on Linux causes child processes to die abruptly or deadlock. `fork` inherits the locked CUDA state. | 2026-01-16 |
| **The IterableDataset Trap** | `IterableDataset` does not implement `__len__`. When using it for online diversification, you MUST manually calculate `max_steps`. | 2026-01-17 |
| **Local vs Hub Dataset Ambiguity** | Relying solely on `load_dataset` can fail in air-gapped or local-first environments. Always implement a check for local directories (e.g., `datasets/name`) before falling back to the HuggingFace Hub. | 2026-01-17 |
| **W&B Auto-detection Breaking** | Hardcoding `report_to="wandb" if settings.wandb_api_key else "none"` breaks automatic W&B detection for users already logged in via CLI or environment. Default to `"wandb"` or `"all"` and let the library handle authentication presence. | 2026-01-17 |
| **Silent CLI Default Overwrites** | When changing a boolean flag (like `--augment-dynamic`) to a toggle (`--augment-dynamic/--no-augment-dynamic`), remember to change the downstream config logic from `if flag: config[key] = True` to `config[key] = flag` to allow users to explicitly disable the new default. | 2026-01-17 |
| **The RAM Explosion Trap** | Loading multiple translation models (Opus-MT) in parallel workers during Dynamic Mode can quickly exceed system RAM (8 models * 300MB * N workers). Maintain a class-level `_model_cache` to share models across worker instances if possible (though some frameworks serialize workers). | 2026-01-17 |
| **Unnecessary W&B Token Check** | Blocking W&B initialization only if a token is present in internal settings prevents using the Public API (which can use cached credentials or environment variables). Always try `wandb.login()` and let the library handle the authentication state. | 2026-01-17 |
| **The Embedding Gradient Trap** | 7B models with large vocabularies (Qwen 151k) have significantly larger embedding layers. Training these with LoRA $r=128$ can consume 4GB+ of extra VRAM just for the adapter weights and optimizer states. Use $r=64$ or $r=32$ for better stability on 24GB GPUs. | 2026-01-17 |
| **The Missing EOS Token Trap** | When finetuning with raw text (no chat template), models never see their EOS token during training. At inference, they generate endlessly because they never learned when to stop. ALWAYS use `tokenizer.apply_chat_template()` or manually append `tokenizer.eos_token`. | 2026-01-17 |
| **Model-Specific Chat Templates** | Different models require different formats: Mistral uses `[INST]...[/INST]`, Qwen uses `<\|im_start\|>...<\|im_end\|>`, Llama3 uses `<\|begin_of_text\|>...`. Store data in neutral format (messages) and apply template at train/inference time. | 2026-01-17 |
| **StoppingCriteria Debugging** | Custom `StoppingCriteria` in transformers must track `prompt_length` to only check generated tokens, not the prompt. Checking the full input causes false positives when stop sequences appear in the prompt. | 2026-01-17 |
| **FormatVariationStep Breaks Parsing** | The `format_variation` augmentation replaces IRCA markers (`### ITERATIVE RESOLUTION CYCLE`) with non-standard variants (`<|ITERATIVE RESOLUTION CYCLE|>`). This breaks `parse_corrected_agent_trace()` in finetuning AND completion extraction in diversity evaluation. Disabled in all config files (ADR-006). | 2026-01-17 |
| **Diversity Evaluation Format Mismatch** | Perplexity computed on IRCA-formatted text doesn't reflect finetuned model performance because the model was trained on chat-template-formatted text. For accurate evaluation, convert to chat template before computing perplexity. | 2026-01-17 |
| **Two Parallel Augmentation Systems** | `src/augmentation/` and `src/formatting/` have **different** step registries. `augmentation` has: translate, shuffle_functions, newline_variation. `formatting` adds: indent_functions. When writing tests, ensure you use step types valid for the specific module being tested. | 2026-01-17 |
| **Running Multiple GPU Tasks in Parallel** | NEVER run multiple GPU-intensive tasks (training, evaluation, inference) concurrently with background processes. The system cannot handle multiple models loaded in GPU memory. Always run sequentially and wait for one to complete before starting another. | 2026-01-17 |
| **JSON Format Mismatch Between Training and Inference** | Training data may use compact JSON (no indentation), but inference code might use pretty-printed JSON (`indent=4`). This causes format mismatch that confuses fine-tuned models. Always use `json.dumps(data, separators=(",", ": "))` to match training format. | 2026-01-17 |
| **Qwen3 vs Qwen2.5 Chat Templates** | Qwen3 models automatically add `<think></think>` tags around reasoning. Qwen2.5 does NOT have these tags. Using the wrong tokenizer causes unexpected output format. Always verify tokenizer matches the actual base model used. | 2026-01-17 |
| **Trusting experiment.json Over adapter_config.json** | The `experiment.json` metadata file can be outdated or incorrect. The authoritative source for the base model is `adapter_config.json` in the model directory (field: `base_model_name_or_path`). Always verify there when debugging format issues. | 2026-01-17 |
| **Missing IRCA System Instructions in Inference** | Server endpoints may default to generic prompts like "You are a helpful assistant" instead of IRCA instructions. Fine-tuned IRCA models require the full IRCA prompt (including example) to produce correctly formatted output. | 2026-01-17 |


---

## Patterns That Work
<!-- Successful approaches to reuse -->

- **CPU-Offloading for Aux Models**: For auxiliary tasks like on-the-fly translation, force the models to run on **CPU**. These models (like Opus-MT) are small (~300MB) and CPU inference is fast enough to prefill the data queue without competing for the VRAM needed by the main 7B+ parameter LLM.
- **Unload-Between-Stages Workflow**: When running complex pipelines (Augment -> Load LLM -> Train), explicitly perform the heavy CPU-bound/Multiprocessing stages *before* initializing the GPU state. This avoids fork conflicts and optimizes total RAM/VRAM footprint.
- **Hybrid Verification**: For features involving complex string manipulation or external model dependencies (like translation), always perform at least one manual end-to-end verification with realistic data. Use this manual pass to "ground" your automated tests.
- **Multilingual Scaling with Qwen-2.5**: When moving beyond 5-8 European languages to mass multilingualism (Asian, Indic, Arabic), prioritize Qwen-2.5 as the base model. Its larger vocabulary (151k) and native multilingual training provide significantly better fluency and reasoning depth compared to Mistral (32k) or Llama (unless usando Llama 3.1+).
- **Group Model Prefixes**: Helsinki-NLP models like `en-inc` or `en-dra` require explicit target language prefixes (e.g., `>>bn<<`) in the input string to function. Always verify the model description for required markers.
- **Local Dataset Loading**: Using `datasets.load_from_disk(path)` when `os.path.exists(path)` allows using locally generated datasets without pushing to Hub.
- **Handling DatasetDict**: Always check if loaded dataset is `DatasetDict` or `Dataset` and extract `train` split if necessary to avoid `KeyError` or type errors in Trainer.
- **Stable QLoRA LR**: For 7B models (like Mistral), a learning rate of `2e-4` is much more stable than `1e-3`.
- **Pre-Diversification for Observability**: If the training process seems "stalled" or you suspect translation issues, disable `--augment-dynamic`. Static mode provides a clear progress bar and isolates translation failures from training logic.
- **Toggle Defaults**: Use Click's `/--no-` syntax for boolean flags. It provides a cleaner CLI (`--no-augment-dynamic`) than negative flags like `--disable-augmentation`.
- **Three-Stage Augmentation Pipeline**: Separating augmentation into structural (before formatting), formatting (during), and presentation (after) stages allows precise control over what gets modified and when. Translation operates on parsed components while format variations operate on final text.
- **Format → Finetune Workflow**: Raw datasets with structured columns (like `corrected_agent_trace`) must be converted to `text` column using `irca dataset format` before finetuning. This decouples augmentation from training.
- **Epoch Scaling with Multiplier**: When using 3x dataset multiplication, reduce epochs proportionally (3 epochs → 1 epoch) to maintain similar total training steps.
- **Reproducible Language Distribution**: Using `translate` step with `probability: 0.67` and `weights: [1, 1]` for two languages achieves approximately 1/3 English (untranslated), 1/3 each translated language.
- **Model-Agnostic Data Storage**: Store training data as structured messages (role/content pairs), not pre-formatted text. Apply `tokenizer.apply_chat_template()` at training time. This ensures each model receives its native format with correct special tokens.
- **Chat Template Validation**: Before finetuning a new model, verify its chat template by calling `tokenizer.apply_chat_template([{"role": "user", "content": "test"}], tokenize=False)` and inspecting the output for expected special tokens.
- **Safe Augmentation Selection**: Prefer augmentations that don't modify parsing markers: `translate`, `shuffle_functions`, `indent_functions`, `newline_variation`. Avoid `format_variation` when using chat templates (breaks parsing).
- **Consistent Format Pipeline**: Training, inference, and evaluation must use the same format. If training applies `tokenizer.apply_chat_template()`, evaluation should too (or perplexity values are misleading).
- **Clean Removal of Deprecated Features**: When removing deprecated augmentation steps (like `format_variation`), remove from: (1) STEP_REGISTRY, (2) config validation, (3) all config files, (4) all tests. Leaving disabled entries creates confusion and potential test failures.
- **Fair Augmentation Comparison**: When comparing augmented vs non-augmented training, control for **training steps** (compute budget), not epochs. If non-augmented has 152 samples and augmented has 436, use ~3 epochs for non-augmented and ~1 epoch for augmented to match step count. This isolates augmentation effect from "more training" effect.
- **Overfitting Diagnostics**: Always report BOTH train and test perplexity. The generalization gap `(test_ppl - train_ppl) / train_ppl` reveals whether model is overfitting. If non-augmented has large gap but augmented doesn't, augmentation helps by preventing overfitting.
- **Checkpoint Analysis for Learning Curves**: Save intermediate checkpoints (e.g., every epoch) to analyze learning dynamics. Comparing train/test loss at each checkpoint reveals when overfitting begins.
- **Verify Base Model via adapter_config.json**: When debugging inference format issues, always check `adapter_config.json` (field: `base_model_name_or_path`) to confirm the actual base model. Don't trust experiment metadata files.
- **Agentic Loop for Multi-Step Function Calling**: When models stop at function calls (due to `<|wait|>` stop sequence), implement an agentic loop: (1) detect function call pattern in output, (2) add mock/real function output, (3) continue generation until FINAL ANSWER or max iterations. This enables complete IRCA cycles.
- **Compact JSON for Training/Inference Consistency**: Use `json.dumps(data, separators=(",", ": "))` (compact, no indent) for function definitions in both training and inference. Pretty-printed JSON with `indent=4` creates a format mismatch that confuses fine-tuned models.
- **IRCA System Instructions with Example**: For fine-tuned IRCA models, the inference prompt must include the full IRCA instructions with a complete example. Without this, models produce generic conversational output instead of the Thought/Action/Call function format.


---

## Patterns That Failed
<!-- Approaches to avoid -->

- **Direct `TrainingArguments` usage with `SFTTrainer`**: `SFTTrainer` now expects `SFTConfig` (which subclasses `TrainingArguments`) for SFT-specific parameters.

---

## Codebase Knowledge
<!-- Facts about this specific codebase -->

- **Model Configuration**: `src/config/settings.py` defines model presets (`mistral`, `mistral-v3`, etc.).
- **CLI Structure**: `src/cli/commands/` contains the implementation for `irca` CLI commands.
- **Finetuning**: Uses `trl` library with QLoRA (`peft`, `bitsandbytes`).
- **Dataset Generation**: Datasets are stored in `datasets/` directory.
- **Frontend**: Located in `ui/`. Built with Vite + React + Tailwind v4. Runs on port 3001 (default).
- **Backend API**: FastAPI app in `src/server/`. Uses `uvicorn`.
- **Session Persistence**: Simple file-based JSON storage (`data/sessions/`) is sufficient for local playground history.
- **Dataset Formatting Module**: `src/formatting/` contains the staged augmentation pipeline. Key files: `config.py` (FormatConfig), `steps.py` (augmentation implementations), `formatter.py` (DatasetFormatter).
- **Diversity Module**: `src/diversity/` contains metrics (lexical, semantic, perplexity). Used via `irca dataset diversity` CLI.
- **Config Files**: Project configs stored in `configs/` directory, including `format_full_augment.json` for standard augmentation.
- **Chat Template Module**: `src/formatting/chat_template.py` handles model-specific formatting. Key functions: `irca_to_messages()`, `apply_chat_template()`, `format_for_inference()`, `get_stop_sequences()`.
- **Diversity Chat Template Support**: `src/diversity/utils.py` contains `convert_irca_to_chat_template()` for evaluation. All perplexity functions accept `apply_chat_template=True` (default).
- **Augmentation Module Separation**: Two modules exist with different purposes:
  - `src/augmentation/` - Simple pipeline augmentation (steps: translate, shuffle_functions, newline_variation)
  - `src/formatting/` - Staged augmentation with structural/formatting/presentation phases (steps: translate, shuffle_functions, indent_functions, newline_variation)
- **Dataset Split Command**: `irca dataset split` creates reproducible train/test splits. Always split BEFORE augmentation to avoid data leakage.
- **Experiment Directory Structure**: `experiments/expNNN-name/` contains train/, test/, train_formatted/, train_augmented/, test_formatted/, models/, results/, experiment.json, README.md.
- **Experiment CLI Commands**: `irca experiment eval-checkpoints` evaluates all checkpoints in a model directory on train/test sets. `irca experiment compare` generates comparison reports with ADR-008 interpretation.
- **Statistics Module**: `src/diversity/statistics.py` provides `bootstrap_confidence_interval()`, `bootstrap_difference_test()`, `effect_size_cohens_d()`, and `interpret_effect_size()`.

---

## Recently Added
<!--
PURPOSE: Track WHAT was added to permanent sections (not duplicate storage)
FORMAT: Brief notes like "Added X to Common Mistakes" or "Updated Codebase Knowledge with Y"
CLEANUP: Remove entries older than 7 days
-->

### Session: Playground & Robustness (2026-01-15)
- Added FastAPI Event Loop blocking mistake (sync GPU calls).
- Added Tailwind v4 migration note.
- Documented Frontend/Backend structure.

### Session: Finetuning Mistral v0.3 (2026-01-15)
- Added `mistral-v3` specific requirements (`sentencepiece`) to Common Mistakes.
- Added notes about `trl.SFTConfig` vs `TrainingArguments` migration.
- Added Local Dataset Loading pattern.

### Session: Format Pipeline & Finetuning (2026-01-16)
- Added three-stage augmentation pipeline pattern to Patterns That Work.
- Added format→finetune workflow pattern.
- Added epoch scaling guidance with multiplier.
- Added reproducible language distribution pattern.
- Updated Codebase Knowledge with formatting and diversity modules.
- Created CLAUDE.md with project skills and user preferences.

### Session: Chat Template Implementation (2026-01-17)
- Added "The Missing EOS Token Trap" to Common Mistakes.
- Added "Model-Specific Chat Templates" to Common Mistakes.
- Added "StoppingCriteria Debugging" to Common Mistakes.
- Added "FormatVariationStep Breaks Parsing" to Common Mistakes.
- Added "Diversity Evaluation Format Mismatch" to Common Mistakes.
- Added "Model-Agnostic Data Storage" to Patterns That Work.
- Added "Chat Template Validation" to Patterns That Work.
- Added "Safe Augmentation Selection" to Patterns That Work.
- Added "Consistent Format Pipeline" to Patterns That Work.
- Created `src/formatting/chat_template.py` with model-specific formatting utilities.
- Updated finetuning and inference to use consistent chat template formatting.
- Created ADR-006: Chat Template Consistency Across Pipeline.
- Disabled `format_variation` in all config files (breaks chat template parsing).

### Session: Chat Template Consistency Phase 2 (2026-01-17)
- Implemented ADR-006 Phase 2: Chat template conversion in diversity evaluation.
- Added `convert_irca_to_chat_template()` and `get_chat_template_completion_markers()` to `src/diversity/utils.py`.
- Added `apply_chat_template=True` parameter to all perplexity/evaluation functions.
- Updated CLI `irca dataset diversity` with `--no-chat-template` flag.
- Removed `format_variation` from STEP_REGISTRY (both `src/formatting/` and `src/augmentation/` modules).
- Cleaned up all config files to remove disabled format_variation entries.
- Updated all unit and integration tests to use valid augmentation steps only.
- Added "Two Parallel Augmentation Systems" to Common Mistakes (lesson from test failure).
- Added "Clean Removal of Deprecated Features" to Patterns That Work.
- Added "Augmentation Module Separation" to Codebase Knowledge.

### Session: ADR-008 Implementation (2026-01-17)
- Implemented all 4 phases of ADR-008 fair comparison methodology tooling.
- Extended `irca finetune run` with: `--output`, `--batch-size`, `--lora-r`, `--save-steps`, `--seed`.
- Created `irca experiment eval-checkpoints` for evaluating all checkpoints in a directory.
- Created `irca experiment compare` for baseline vs augmented comparison with interpretation.
- Created `src/diversity/statistics.py` with `bootstrap_confidence_interval()` and `effect_size_cohens_d()`.
- Added `generate_experiment_comparison_report()` to `src/diversity/report.py`.
- All 19 unit tests passing for statistics module.

### Session: Experimental Framework for Augmentation (2026-01-17)
- Created ADR-007: Experimental Framework for Augmentation Impact Analysis.
- Created ADR-008: Fair Comparison Methodology (same steps, not epochs).
- Implemented `irca dataset split` command for reproducible train/test splits.
- Created experiment structure: `experiments/exp001-augmentation-impact/`.
- Added "Fair Augmentation Comparison" to Patterns That Work.
- Added "Overfitting Diagnostics" to Patterns That Work.
- Added "Checkpoint Analysis" to Patterns That Work.
- Added experiment directory structure to Codebase Knowledge.

### Session: Experiment Execution (2026-01-17)
- Executed exp001-augmentation-impact experiment end-to-end.
- Trained baseline model (3 epochs, 114 steps) and augmented model (1 epoch, 109 steps).
- Evaluated on both English-only test set (38 samples) and augmented test set (111 samples).
- **Results**: Augmentation does NOT improve test perplexity. Augmented model is 6.7% worse on English-only test. Both perform similarly on multilingual test.
- Added "Running Multiple GPU Tasks in Parallel" to Common Mistakes (lesson from OOM when running concurrent evaluations).
- Updated CLAUDE.md with GPU Memory Limitation critical constraint.

### Session: Playground Inference Debugging (2026-01-17)
- Debugged IRCA model not producing correct format in playground.
- Root cause: (1) Missing IRCA system instructions, (2) JSON format mismatch (pretty vs compact).
- Fixed `src/server/routers/generation.py` with `DEFAULT_IRCA_INSTRUCTIONS` constant.
- Changed JSON formatting from `indent=4` to `separators=(",", ": ")` for compact format.
- Implemented agentic loop for multi-step function calling with mock outputs.
- Added 4 new entries to Common Mistakes: JSON format mismatch, Qwen3 vs Qwen2.5 templates, trusting experiment.json, missing IRCA instructions.
- Added 4 new entries to Patterns That Work: verify via adapter_config, agentic loop, compact JSON, IRCA instructions with example.
