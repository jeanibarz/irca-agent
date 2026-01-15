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

**Last Updated:** 2026-01-15 (Finetuning Session)

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

---

## Patterns That Work
<!-- Successful approaches to reuse -->

- **Online Data Diversification**: Using `IterableDataset` with `.map(batched=True)` and `num_workers > 0` in the `DataLoader` allows performing heavy transformations (like translation) without bottlenecking the GPU training.
- **Class-Level Model Caching**: When using multiprocessing for data loading, class-level caches for local models ensure each worker only loads the model once, reducing startup overhead.
- **Reference Highlighting**: Delimiting tool citations in final answers allows for clean UI visualization while maintaining raw text readability.
- **Local Dataset Loading**: Using `datasets.load_from_disk(path)` when `os.path.exists(path)` allows using locally generated datasets without pushing to Hub.
- **Handling DatasetDict**: Always check if loaded dataset is `DatasetDict` or `Dataset` and extract `train` split if necessary to avoid `KeyError` or type errors in Trainer.
- **Stable QLoRA LR**: For 7B models (like Mistral), a learning rate of `2e-4` is much more stable than `1e-3`.

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
