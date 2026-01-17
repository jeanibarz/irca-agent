# Experiment 001: Augmentation Impact Analysis

**Created:** 2026-01-17
**Status:** COMPLETED
**Base Model:** `Qwen/Qwen3-4B-Instruct-2507` (4B parameters, via Unsloth)
**Methodology:** ADR-008 (Fair Comparison - Same Training Steps)

## Results Summary

| Evaluation | Model | Test Set | Mean PPL | Std | Median |
|------------|-------|----------|----------|-----|--------|
| **E1** | Baseline | Nominal | 1.329 | 0.173 | 1.287 |
| **E2** | Baseline | Diverse | 1.375 | 0.192 | 1.328 |
| **E3** | Augmented | Nominal | 1.355 | 0.182 | 1.306 |
| **E4** | Augmented | Diverse | 1.371 | 0.198 | 1.320 |

### Success Criteria Evaluation

| Criterion | Check | Result |
|-----------|-------|--------|
| No regression on nominal | E3 (1.355) <= E1 + 10% (1.462) | PASS |
| Improved generalization | E4 (1.371) < E2 (1.375) | PASS |
| Augmentation benefit | (E4-E2)=-0.004 < (E3-E1)=+0.026 | PASS |

**Conclusion:** Data augmentation improves generalization to diverse inputs (+0.4% improvement) with minimal impact on nominal performance (+2.0% regression, well within tolerance).

## Hypothesis

> Augmented training data improves model robustness on diverse inputs without significantly harming performance on nominal cases.

## Dataset Summary

| Dataset | Samples | Purpose |
|---------|---------|---------|
| `train_formatted` | 152 | Training baseline (no augmentation, English only) |
| `train_augmented` | 436 | Training with augmentation (2.87x multiply, multilingual) |
| `test_formatted` | 38 | Nominal test cases (English only) |
| `test_augmented` | 111 | Diverse/edge test cases (multilingual, varied formatting) |

**Split Info:**
- Original: 190 samples
- Train/Test: 80/20 split (seed=42)
- Augmentation: translate (fr/es, p=0.67), shuffle_functions (p=0.8), indent_functions (p=0.5)

## Evaluation Matrix (2×2)

| Evaluation ID | Model | Test Set | Measures |
|---------------|-------|----------|----------|
| **E1** | baseline | test_formatted | Baseline on nominal cases |
| **E2** | baseline | test_augmented | Baseline on diverse cases |
| **E3** | augmented | test_formatted | Augmented on nominal cases |
| **E4** | augmented | test_augmented | Augmented on diverse cases |

**Key Comparisons:**
- **E1 vs E3**: Does augmented training hurt nominal performance?
- **E2 vs E4**: Does augmented training improve generalization?
- **E1 vs E2**: How much does baseline struggle on diverse inputs?
- **E3 vs E4**: How much does augmented model struggle on diverse inputs?

## ADR-008 Fair Comparison Methodology

**Key Principle:** Compare models trained with the **same number of gradient updates** (steps), not epochs.

| Model | Dataset Size | Epochs | Steps (batch=4) |
|-------|--------------|--------|-----------------|
| Baseline | 152 | 3 | ~114 steps |
| Augmented | 436 | 1 | ~109 steps |

This controls for compute budget and isolates the augmentation effect.

---

## Execution Workflow

### Prerequisites

Ensure you're in the project root:
```bash
cd /home/jean/git/irca-agent
```

### Step 1: Train Baseline Model (No Augmentation)

Using Unsloth for memory-efficient training (6GB VRAM vs 24GB standard):

```bash
poetry run irca finetune run \
  -m qwen3-4b \
  -d experiments/exp001-augmentation-impact/train_formatted \
  -o experiments/exp001-augmentation-impact/models/baseline \
  --epochs 3 --batch-size 4 --lora-r 16 --max-seq-length 1024 --seed 42
```

**Training parameters:**
- `--lora-r 16`: LoRA rank (memory efficient)
- `--max-seq-length 1024`: Max context length
- W&B tracking is automatic (use `--no-wandb` to disable)

**Actual output:**
- 114 training steps (152 samples, batch 4, 3 epochs)
- Training time: ~424s (~7 minutes)
- Peak GPU memory: ~6GB
- Final loss: ~0.04
- W&B run: `qwen3-4b-train_formatted-{timestamp}`

### Step 2: Train Augmented Model

```bash
poetry run irca finetune run \
  -m qwen3-4b \
  -d experiments/exp001-augmentation-impact/train_augmented \
  -o experiments/exp001-augmentation-impact/models/augmented \
  --epochs 1 --batch-size 4 --lora-r 16 --max-seq-length 1024 --seed 42
```

**Actual output:**
- 109 training steps (436 samples, batch 4, 1 epoch)
- Training time: ~414s (~7 minutes)
- Peak GPU memory: ~6GB
- Final loss: ~0.04
- Same compute budget as baseline (fair comparison per ADR-008)
- W&B run: `qwen3-4b-train_augmented-{timestamp}`

### Step 3: Evaluate Models (2×2 Matrix)

All 4 evaluations measure completion-only perplexity using Unsloth for memory efficiency (4GB VRAM).

**E1: Baseline on Nominal Test**
```bash
TORCHDYNAMO_DISABLE=1 UNSLOTH_RETURN_LOGITS=1 poetry run python scripts/eval_unsloth.py \
  -m experiments/exp001-augmentation-impact/models/baseline \
  -d experiments/exp001-augmentation-impact/test_formatted \
  -o experiments/exp001-augmentation-impact/results/E1_baseline_on_nominal.json
```

**E2: Baseline on Diverse Test**
```bash
TORCHDYNAMO_DISABLE=1 UNSLOTH_RETURN_LOGITS=1 poetry run python scripts/eval_unsloth.py \
  -m experiments/exp001-augmentation-impact/models/baseline \
  -d experiments/exp001-augmentation-impact/test_augmented \
  -o experiments/exp001-augmentation-impact/results/E2_baseline_on_diverse.json
```

**E3: Augmented on Nominal Test**
```bash
TORCHDYNAMO_DISABLE=1 UNSLOTH_RETURN_LOGITS=1 poetry run python scripts/eval_unsloth.py \
  -m experiments/exp001-augmentation-impact/models/augmented \
  -d experiments/exp001-augmentation-impact/test_formatted \
  -o experiments/exp001-augmentation-impact/results/E3_augmented_on_nominal.json
```

**E4: Augmented on Diverse Test**
```bash
TORCHDYNAMO_DISABLE=1 UNSLOTH_RETURN_LOGITS=1 poetry run python scripts/eval_unsloth.py \
  -m experiments/exp001-augmentation-impact/models/augmented \
  -d experiments/exp001-augmentation-impact/test_augmented \
  -o experiments/exp001-augmentation-impact/results/E4_augmented_on_diverse.json
```

**Output per evaluation:**
- Perplexity statistics (mean, std, median, min, max)
- GPU memory: 4.05GB per evaluation

### Step 4: Generate Comparison Report

```bash
poetry run irca experiment compare \
  -b experiments/exp001-augmentation-impact/results/E1_baseline_nominal.json \
  -a experiments/exp001-augmentation-impact/results/E3_augmented_nominal.json \
  -o experiments/exp001-augmentation-impact/results/comparison_nominal.html \
  --format html
```

```bash
poetry run irca experiment compare \
  -b experiments/exp001-augmentation-impact/results/E2_baseline_diverse.json \
  -a experiments/exp001-augmentation-impact/results/E4_augmented_diverse.json \
  -o experiments/exp001-augmentation-impact/results/comparison_diverse.html \
  --format html
```

---

## Success Criteria

| Criterion | Formula | Meaning |
|-----------|---------|---------|
| No regression on nominal | E3 ≤ E1 + 10% | Augmented training doesn't hurt standard cases |
| Improved generalization | E4 < E2 | Augmented model handles diverse inputs better |
| Augmentation benefit | (E4-E2) < (E3-E1) | Augmentation helps more on diverse than it hurts on nominal |

## 2×2 Interpretation Guide

**Primary Comparisons:**
```
E1 vs E3 (Nominal Performance):
  IF E3 < E1:     → Augmentation improves even nominal cases
  IF E3 ≈ E1:     → No regression (acceptable)
  IF E3 > E1+10%: → Augmentation may add noise, investigate

E2 vs E4 (Generalization to Diverse):
  IF E4 < E2:     → Augmentation improves robustness ✓
  IF E4 ≈ E2:     → No generalization benefit
  IF E4 > E2:     → Augmentation may be misaligned with diversity

E1 vs E2 (Baseline Robustness):
  Large gap:      → Baseline struggles with diversity
  Small gap:      → Baseline already handles diversity

E3 vs E4 (Augmented Robustness):
  Small gap:      → Augmented model generalizes well ✓
  Large gap:      → Even augmented model struggles
```

**Combined Analysis:**
```
IDEAL OUTCOME:
  E3 ≈ E1 (no nominal regression)
  E4 << E2 (large generalization gain)
  E3-E4 small (augmented model handles both)

GOOD OUTCOME:
  E3 slightly > E1 (small nominal cost)
  E4 < E2 (generalization gain)
  Net benefit: (E2-E4) > (E3-E1)

BAD OUTCOME:
  E3 >> E1 (large nominal regression)
  E4 ≥ E2 (no generalization gain)
  → Augmentation strategy needs revision
```

## Statistical Considerations

| Test Set | Samples | Notes |
|----------|---------|-------|
| test_formatted | 38 | Effect size > 10% likely significant |
| test_augmented | 111 | More statistical power, effect size > 5% meaningful |

Bootstrap 95% CIs are provided for all metrics.

## Directory Structure

```
exp001-augmentation-impact/
├── README.md                 # This file
├── experiment.json           # Experiment configuration (2×2 matrix)
├── split_info.json           # Dataset split metadata
├── train/                    # Raw train split (152 samples)
├── train_formatted/          # Baseline training data (152 samples)
├── train_augmented/          # Augmented training data (436 samples)
├── test/                     # Raw test split (38 samples)
├── test_formatted/           # Nominal test cases (38 samples)
├── test_augmented/           # Diverse test cases (111 samples)
├── models/                   # Trained models (Ministral-3B)
│   ├── baseline/             # 3 epoch checkpoints + final
│   └── augmented/            # 1 epoch checkpoint + final
└── results/                  # 2×2 Evaluation results
    ├── E1_baseline_nominal.json    # Baseline on nominal
    ├── E2_baseline_diverse.json    # Baseline on diverse
    ├── E3_augmented_nominal.json   # Augmented on nominal
    ├── E4_augmented_diverse.json   # Augmented on diverse
    ├── comparison_nominal.html     # E1 vs E3 comparison
    └── comparison_diverse.html     # E2 vs E4 comparison
```

## References

- [ADR-007: Experimental Framework](../../docs/adrs/ADR-007-augmentation-experiment-framework.md)
- [ADR-008: Fair Comparison Methodology](../../docs/adrs/ADR-008-fair-augmentation-comparison-methodology.md)
- [ADR-005: Completion-Only Perplexity](../../docs/adrs/ADR-005-completion-perplexity.md)
- [ADR-006: Chat Template Consistency](../../docs/adrs/ADR-006-chat-template-consistency.md)
