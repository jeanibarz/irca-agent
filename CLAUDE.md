# CLAUDE.md - Project Skills and Preferences

## User Preferences

### Default Dataset
- **Raw dataset**: `/home/jean/git/irca-agent/datasets/irca_agent_dataset_v5-5acc`
- **Formatted output naming**: `{dataset_name}-formatted-augmented`

### Default Model for Finetuning
- **Model**: `qwen3-4b` (Qwen/Qwen3-4B)
- **Default epochs**: 1 (when using 3x multiplied dataset)

### Default Augmentation Configuration
- **Multiplier**: 3x
- **Languages**: French (fr), Spanish (es), English (en) - approximately 1/3 each
- **Translation probability**: 0.67 (2/3 translated, 1/3 stays English)
- **All augmentation features enabled**:
  - Translation (structural): query + final answer
  - Function shuffling (80%)
  - JSON indentation variation (50%)
  - Format variation (30%)
  - Newline variation (20%)

### Config File Location
- **Format config**: `/home/jean/git/irca-agent/configs/format_full_augment.json`

---

## Skills / Workflows

### Skill: Format Dataset with Augmentation

**Purpose**: Convert raw structured dataset to training-ready format with multilingual diversification.

**Command**:
```bash
poetry run irca dataset format \
  -i datasets/irca_agent_dataset_v5-5acc \
  -o datasets/irca_agent_dataset_v5-5acc-formatted-augmented \
  -c configs/format_full_augment.json \
  --seed 42
```

**Config template** (`configs/format_full_augment.json`):
```json
{
  "seed": 42,
  "multiply": 3,
  "deduplicate": true,
  "source_column": "corrected_agent_trace",
  "structural": [
    {
      "type": "translate",
      "probability": 0.67,
      "params": {
        "languages": ["fr", "es"],
        "weights": [1, 1],
        "targets": ["query", "answer"]
      }
    }
  ],
  "formatting": [
    {"type": "shuffle_functions", "probability": 0.8},
    {"type": "indent_functions", "probability": 0.5, "params": {"choices": [null, 2, 4]}}
  ],
  "presentation": [
    {"type": "format_variation", "probability": 0.3},
    {"type": "newline_variation", "probability": 0.2}
  ]
}
```

**Output**:
- Dataset with `text` column ready for finetuning
- Language distribution: ~33% English, ~33% French, ~33% Spanish
- Various augmentation combinations applied

---

### Skill: Format Dataset (Baseline, No Augmentation)

**Purpose**: Create formatted baseline dataset without augmentations for diversity comparison.

**Command**:
```bash
poetry run irca dataset format \
  -i datasets/irca_agent_dataset_v5-5acc \
  -o datasets/irca_agent_dataset_v5-5acc-formatted-baseline \
  --no-augment \
  --seed 42
```

---

### Skill: Diversity Evaluation (Single Dataset)

**Purpose**: Analyze diversity metrics of a single formatted dataset.

**Command**:
```bash
# Quick mode (no model loading, lexical metrics only)
poetry run irca dataset diversity -d datasets/irca_agent_dataset_v5-5acc-formatted-augmented --quick

# Full mode (includes perplexity, requires GPU)
poetry run irca dataset diversity -d datasets/irca_agent_dataset_v5-5acc-formatted-augmented -m gpt2
```

**Metrics reported**:
- Distinct-1, Distinct-2, Distinct-3 (lexical diversity)
- Vendi Score (semantic diversity)
- Perplexity stats (if model specified)

---

### Skill: Diversity Comparison (Original vs Augmented)

**Purpose**: Compare diversity between baseline and augmented datasets.

**Prerequisites**: Both datasets must be formatted (have `text` column).

**Commands**:
```bash
# 1. Create baseline (if not exists)
poetry run irca dataset format \
  -i datasets/irca_agent_dataset_v5-5acc \
  -o datasets/baseline \
  --no-augment

# 2. Create augmented (if not exists)
poetry run irca dataset format \
  -i datasets/irca_agent_dataset_v5-5acc \
  -o datasets/augmented \
  -c configs/format_full_augment.json

# 3. Compare
poetry run irca dataset diversity \
  -o datasets/baseline \
  -a datasets/augmented \
  --quick
```

**Output**: Side-by-side comparison with delta percentages.

---

### Skill: Finetune Model

**Purpose**: Train model using Q-LoRA on formatted dataset.

**Command**:
```bash
poetry run irca finetune run \
  -m qwen3-4b \
  -d datasets/irca_agent_dataset_v5-5acc-formatted-augmented \
  --epochs 1
```

**Available models**: `mistral`, `mistral-v3`, `tinyllama`, `qwen-7b`, `qwen-4b`, `qwen-14b`, `qwen3-8b`, `qwen3-4b`

**Output**:
- Model saved to: `/home/jean/git/irca-agent/models/finetuned_models/Qwen3-4B_irca_agent_v5-6/`
- W&B tracking: https://wandb.ai/ibarz-jean-home/irca-agent

**Epoch guidance**:
- 1 epoch for 3x multiplied dataset (~550 samples)
- 3 epochs for base dataset (~190 samples)

---

### Skill: Inspect Dataset

**Purpose**: View dataset structure and sample entries.

**Command**:
```bash
poetry run irca dataset inspect datasets/irca_agent_dataset_v5-5acc
```

---

### Skill: Full Workflow (Format + Finetune)

**Purpose**: Complete pipeline from raw dataset to finetuned model.

**Commands**:
```bash
# Step 1: Format with augmentation
poetry run irca dataset format \
  -i datasets/irca_agent_dataset_v5-5acc \
  -o datasets/irca_agent_dataset_v5-5acc-formatted-augmented \
  -c configs/format_full_augment.json

# Step 2: Verify language distribution (optional)
poetry run python -c "
import datasets
ds = datasets.load_from_disk('datasets/irca_agent_dataset_v5-5acc-formatted-augmented')
print(f'Total samples: {len(ds)}')
fr = sum(1 for a in ds['augmentations'] if 'translate:fr' in a)
es = sum(1 for a in ds['augmentations'] if 'translate:es' in a)
en = len(ds) - fr - es
print(f'English: {en} ({100*en/len(ds):.1f}%)')
print(f'French: {fr} ({100*fr/len(ds):.1f}%)')
print(f'Spanish: {es} ({100*es/len(ds):.1f}%)')
"

# Step 3: Finetune
poetry run irca finetune run \
  -m qwen3-4b \
  -d datasets/irca_agent_dataset_v5-5acc-formatted-augmented \
  --epochs 1
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Format dataset (augmented) | `irca dataset format -i INPUT -o OUTPUT -c configs/format_full_augment.json` |
| Format dataset (baseline) | `irca dataset format -i INPUT -o OUTPUT --no-augment` |
| Diversity (single) | `irca dataset diversity -d DATASET --quick` |
| Diversity (compare) | `irca dataset diversity -o ORIGINAL -a AUGMENTED --quick` |
| Finetune | `irca finetune run -m qwen3-4b -d DATASET --epochs 1` |
| Inspect dataset | `irca dataset inspect DATASET` |

---

## Key Files

| Purpose | Path |
|---------|------|
| Raw dataset | `datasets/irca_agent_dataset_v5-5acc` |
| Augmentation config | `configs/format_full_augment.json` |
| Finetuned models | `models/finetuned_models/` |
| Format implementation | `src/formatting/` |
| Diversity implementation | `src/diversity/` |
| CLI commands | `src/cli/commands/dataset.py`, `src/cli/commands/finetune.py` |

---

## Important Notes

1. **Dataset must have `text` column for finetuning** - use `irca dataset format` to create it
2. **Translation loads Helsinki-NLP models** - first run downloads ~500MB per language
3. **Finetuning requires GPU** - uses Q-LoRA with 4-bit quantization
4. **W&B tracking is automatic** - logs to https://wandb.ai/ibarz-jean-home/irca-agent
5. **Seed 42 is default** for reproducibility

---

## Critical Constraints

### GPU Memory Limitation

**NEVER run multiple GPU-intensive tasks concurrently.** The system cannot handle multiple models loaded in GPU memory at the same time.

**What this means:**
- Do NOT run model training in parallel with another training job
- Do NOT run multiple checkpoint evaluations concurrently
- Do NOT run model evaluation while training is in progress
- Always wait for one GPU task to complete before starting another

**Correct approach:**
```bash
# Run first evaluation, wait for completion
poetry run irca experiment eval-checkpoints -c models/baseline ...

# Only after the above completes, run the second
poetry run irca experiment eval-checkpoints -c models/augmented ...
```

**Incorrect approach:**
```bash
# DO NOT run these in parallel (background &)
poetry run irca experiment eval-checkpoints -c models/baseline ... &
poetry run irca experiment eval-checkpoints -c models/augmented ... &
```

This applies to any operation that loads a model into GPU memory (finetuning, evaluation, inference).
