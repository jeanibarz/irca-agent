# Implementation Plan: ADR-008 Fair Comparison Methodology

## Objective

Implement the tooling required to execute the ADR-008 methodology for comparing augmented vs. non-augmented training with scientific rigor.

---

## Gap Analysis

### Current State

| Component | Status | Details |
|-----------|--------|---------|
| Dataset split | ✅ Done | `irca dataset split` exists |
| Format datasets | ✅ Done | `irca dataset format` exists |
| Basic finetuning | ✅ Done | `irca finetune run` exists |
| Epoch checkpoints | ✅ Done | `save_strategy="epoch"` |
| Perplexity evaluation | ✅ Done | `evaluate_generalization()` exists |
| Multi-model comparison | ✅ Done | `compare_models_on_dataset()` exists |

### Missing for ADR-008

| Requirement | Gap | Priority |
|-------------|-----|----------|
| Custom output directory | CLI lacks `--output` | High |
| Step-based checkpoints | CLI lacks `--save-steps` | High |
| Batch size control | CLI lacks `--batch-size` | High |
| LoRA rank control | CLI lacks `--lora-r` | High |
| Reproducibility | CLI lacks `--seed` | High |
| Checkpoint evaluation | No batch checkpoint eval | Medium |
| Bootstrap CI | Not implemented | Medium |
| Experiment report | No structured output | Low |

---

## Implementation Tasks

### Phase 1: Extend `irca finetune run` CLI (High Priority)

**File:** `src/cli/commands/finetune.py`

Add the following CLI arguments to the `run` command:

```python
@click.option("--output", "-o", type=str, help="Output directory for model")
@click.option("--batch-size", "-bs", type=int, default=4, help="Effective batch size")
@click.option("--lora-r", type=int, default=64, help="LoRA rank")
@click.option("--save-steps", type=int, help="Save checkpoint every N steps (overrides epoch strategy)")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility")
```

**Changes to training logic:**

1. **Output directory override:**
   ```python
   if output:
       config["output_dir"] = output
   ```

2. **Batch size with gradient accumulation:**
   ```python
   # Calculate accumulation steps for target effective batch size
   per_device_batch = 1  # Keep at 1 for memory
   gradient_accumulation = batch_size  # e.g., 4 → accum=4
   ```

3. **Save strategy:**
   ```python
   if save_steps:
       save_strategy = "steps"
       save_steps_val = save_steps
   else:
       save_strategy = "epoch"
       save_steps_val = None
   ```

4. **Seed setting:**
   ```python
   import transformers
   transformers.set_seed(seed)
   ```

**Estimated changes:** ~40 lines

---

### Phase 2: Add `irca experiment eval` Command (Medium Priority)

**File:** `src/cli/commands/experiment.py` (new file)

Create a new command group for experiment evaluation:

```python
@click.group()
def experiment():
    """Experiment evaluation commands."""
    pass

@experiment.command("eval-checkpoints")
@click.option("--checkpoints-dir", "-c", required=True, help="Directory containing checkpoints")
@click.option("--train-set", required=True, help="Training set for train perplexity")
@click.option("--test-set", required=True, help="Test set for test perplexity")
@click.option("--baseline-model", "-b", help="Baseline model for comparison")
@click.option("--output", "-o", help="Output JSON file for results")
@click.option("--eval-mode", default="completion", help="Evaluation mode")
```

**Logic:**

1. Discover all `checkpoint-*` directories in `--checkpoints-dir`
2. For each checkpoint:
   - Load model
   - Compute train perplexity
   - Compute test perplexity
   - Compute generalization gap
   - Unload model (free VRAM)
3. Output results as JSON + summary table

**Estimated changes:** ~150 lines

---

### Phase 3: Bootstrap Confidence Intervals (Medium Priority)

**File:** `src/diversity/statistics.py` (new file)

```python
def bootstrap_confidence_interval(
    values: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for mean.

    Returns:
        (mean, lower_ci, upper_ci)
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    values = np.array(values)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))

    lower = np.percentile(means, (1 - confidence) / 2 * 100)
    upper = np.percentile(means, (1 + confidence) / 2 * 100)

    return float(np.mean(values)), float(lower), float(upper)
```

**Integration:**

Update `compute_perplexity_profile()` to optionally compute CIs:

```python
def compute_perplexity_profile(..., compute_ci: bool = False):
    ...
    if compute_ci:
        mean, lower, upper = bootstrap_confidence_interval(perplexities)
        result["ci_95_lower"] = lower
        result["ci_95_upper"] = upper
```

**Estimated changes:** ~60 lines

---

### Phase 4: Experiment Report Generation (Low Priority)

**File:** `src/diversity/report.py` (extend existing)

Add function to generate ADR-008 comparison report:

```python
def generate_experiment_report(
    baseline_results: dict,
    augmented_results: dict,
    output_path: str,
    format: str = "html",  # or "json", "markdown"
):
    """
    Generate experiment comparison report per ADR-008 methodology.

    Includes:
    - Training configuration comparison
    - Per-checkpoint metrics table
    - Train/test perplexity curves (if HTML)
    - Generalization gap analysis
    - Statistical significance (bootstrap CIs)
    - Interpretation per ADR-008 decision tree
    """
```

**Estimated changes:** ~200 lines (for HTML report)

---

## Implementation Order

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: CLI Extensions (~40 lines)                             │
│ ├── Add --output, --batch-size, --lora-r, --save-steps, --seed │
│ └── Update SFTConfig generation logic                           │
├─────────────────────────────────────────────────────────────────┤
│ Phase 2: Checkpoint Evaluation Command (~150 lines)             │
│ ├── Create src/cli/commands/experiment.py                       │
│ ├── Implement eval-checkpoints command                          │
│ └── Register in CLI                                             │
├─────────────────────────────────────────────────────────────────┤
│ Phase 3: Bootstrap CI (~60 lines)                               │
│ ├── Create src/diversity/statistics.py                          │
│ └── Integrate with perplexity profile                           │
├─────────────────────────────────────────────────────────────────┤
│ Phase 4: Experiment Report (~200 lines)                         │
│ └── HTML/Markdown report generation                             │
└─────────────────────────────────────────────────────────────────┘

Total estimated: ~450 lines of new/modified code
```

---

## Testing Plan

### Phase 1 Tests

```bash
# Test new CLI arguments
poetry run irca finetune run --help  # Verify new args appear

# Dry-run with small dataset (2 samples, 1 step)
poetry run irca finetune run \
  -m tinyllama \
  -d experiments/exp001-augmentation-impact/train_formatted \
  -o /tmp/test-checkpoint \
  --batch-size 4 \
  --lora-r 32 \
  --save-steps 1 \
  --epochs 1 \
  --seed 42
```

### Phase 2 Tests

```bash
# Test checkpoint evaluation
poetry run irca experiment eval-checkpoints \
  -c /tmp/test-checkpoint \
  --train-set experiments/exp001-augmentation-impact/train_formatted \
  --test-set experiments/exp001-augmentation-impact/test_formatted \
  -o /tmp/results.json
```

### Integration Test

Run full ADR-008 workflow on tiny subset:

```bash
# Split 10 samples: 8 train, 2 test
# Format baseline and augmented
# Train both for 2 steps
# Evaluate checkpoints
# Generate report
```

---

## ADR-008 Workflow After Implementation

```bash
# 1. Train baseline (3 epochs, ~114 steps)
poetry run irca finetune run \
  --model-type mistral-v3 \
  --dataset experiments/exp001-augmentation-impact/train_formatted \
  --output experiments/exp001-augmentation-impact/models/baseline \
  --epochs 3 \
  --batch-size 4 \
  --lora-r 64 \
  --save-steps 38 \
  --seed 42

# 2. Train augmented (1 epoch, ~109 steps)
poetry run irca finetune run \
  --model-type mistral-v3 \
  --dataset experiments/exp001-augmentation-impact/train_augmented \
  --output experiments/exp001-augmentation-impact/models/augmented \
  --epochs 1 \
  --batch-size 4 \
  --lora-r 64 \
  --save-steps 38 \
  --seed 42

# 3. Evaluate all checkpoints
poetry run irca experiment eval-checkpoints \
  -c experiments/exp001-augmentation-impact/models/baseline \
  --train-set experiments/exp001-augmentation-impact/train_formatted \
  --test-set experiments/exp001-augmentation-impact/test_formatted \
  -o experiments/exp001-augmentation-impact/results/baseline.json

poetry run irca experiment eval-checkpoints \
  -c experiments/exp001-augmentation-impact/models/augmented \
  --train-set experiments/exp001-augmentation-impact/train_augmented \
  --test-set experiments/exp001-augmentation-impact/test_formatted \
  -o experiments/exp001-augmentation-impact/results/augmented.json

# 4. Generate comparison report
poetry run irca experiment report \
  --baseline experiments/exp001-augmentation-impact/results/baseline.json \
  --augmented experiments/exp001-augmentation-impact/results/augmented.json \
  -o experiments/exp001-augmentation-impact/results/comparison.html
```

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/cli/commands/finetune.py` | Modify | Add CLI args |
| `src/cli/commands/experiment.py` | Create | New command group |
| `src/cli/main.py` | Modify | Register experiment commands |
| `src/diversity/statistics.py` | Create | Bootstrap CI functions |
| `src/diversity/perplexity.py` | Modify | Add CI computation |
| `src/diversity/report.py` | Modify | Add experiment report |
| `tests/unit/test_statistics.py` | Create | Test bootstrap CI |
| `tests/integration/test_experiment_cli.py` | Create | Test experiment commands |

---

## Acceptance Criteria

- [x] `irca finetune run` accepts `--output`, `--batch-size`, `--lora-r`, `--save-steps`, `--seed`
- [x] Step-based checkpoints are saved when `--save-steps` is provided
- [x] `irca experiment eval-checkpoints` evaluates all checkpoints in a directory
- [x] Perplexity results include 95% bootstrap confidence intervals
- [x] Comparison report shows ADR-008 interpretation guidance
- [x] Full ADR-008 workflow can be executed with documented commands

## Implementation Status: COMPLETE

All phases implemented on 2026-01-17.
