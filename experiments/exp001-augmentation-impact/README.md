# Experiment 001: Augmentation Impact Analysis

**Created:** 2026-01-17
**Status:** Ready to execute
**Methodology:** ADR-008 (Fair Comparison - Same Training Steps)

## Hypothesis

> Data augmentation (translation to French/Spanish, function shuffling) improves model robustness without significantly harming performance on the original test distribution.

## Dataset Summary

| Dataset | Samples | Purpose |
|---------|---------|---------|
| `train_formatted` | 152 | Training baseline (no augmentation) |
| `train_augmented` | 436 | Training with augmentation (2.87x multiply) |
| `test_formatted` | 38 | Held-out evaluation set |

**Split Info:**
- Original: 190 samples
- Train/Test: 80/20 split (seed=42)
- Augmentation: translate (fr/es, p=0.67), shuffle_functions (p=0.8), indent_functions (p=0.5)

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

```bash
poetry run irca finetune run \
  -m mistral-v3 \
  -d experiments/exp001-augmentation-impact/train_formatted \
  -o experiments/exp001-augmentation-impact/models/baseline \
  --epochs 3 \
  --batch-size 4 \
  --lora-r 64 \
  --save-steps 38 \
  --seed 42
```

**Expected output:**
- Checkpoints at steps 38, 76, 114 (one per epoch)
- Final model in `models/baseline/`
- W&B logging to `irca-agent` project

### Step 2: Train Augmented Model

```bash
poetry run irca finetune run \
  -m mistral-v3 \
  -d experiments/exp001-augmentation-impact/train_augmented \
  -o experiments/exp001-augmentation-impact/models/augmented \
  --epochs 1 \
  --batch-size 4 \
  --lora-r 64 \
  --save-steps 38 \
  --seed 42
```

**Note:** 1 epoch for augmented (~109 steps) ≈ 3 epochs for baseline (~114 steps)

### Step 3: Evaluate Baseline Checkpoints

```bash
poetry run irca experiment eval-checkpoints \
  -c experiments/exp001-augmentation-impact/models/baseline \
  --train-set experiments/exp001-augmentation-impact/train_formatted \
  --test-set experiments/exp001-augmentation-impact/test_formatted \
  -o experiments/exp001-augmentation-impact/results/baseline.json
```

**Output includes:**
- Train perplexity for each checkpoint
- Test perplexity for each checkpoint
- 95% bootstrap confidence intervals
- Generalization gap analysis
- Best checkpoint identification

### Step 4: Evaluate Augmented Checkpoints

```bash
poetry run irca experiment eval-checkpoints \
  -c experiments/exp001-augmentation-impact/models/augmented \
  --train-set experiments/exp001-augmentation-impact/train_augmented \
  --test-set experiments/exp001-augmentation-impact/test_formatted \
  -o experiments/exp001-augmentation-impact/results/augmented.json
```

### Step 5: Generate Comparison Report

```bash
poetry run irca experiment compare \
  -b experiments/exp001-augmentation-impact/results/baseline.json \
  -a experiments/exp001-augmentation-impact/results/augmented.json \
  -o experiments/exp001-augmentation-impact/results/comparison.html \
  --format html
```

This generates an HTML report with:
- Side-by-side checkpoint comparison
- Statistical significance analysis
- ADR-008 interpretation guidance

---

## Expected Outcomes

| Outcome | Test PPL Change | Interpretation | Action |
|---------|-----------------|----------------|--------|
| Significant improvement | > 10% lower | Augmentation works well | Expand augmentation strategy |
| Modest improvement | 5-10% lower | Marginal benefit | Keep if preprocessing is cheap |
| No difference | Within 5% | Augmentation neutral | Consider collecting more data |
| Worse | > 5% higher | Augmentation hurts | Investigate problematic augmentations |

## ADR-008 Interpretation Guide

```
IF aug_test_ppl < baseline_test_ppl:
    IF baseline has large generalization gap:
        → Augmentation prevents overfitting
    ELSE:
        → Augmentation improves learning

IF aug_test_ppl ≈ baseline_test_ppl:
    → No significant effect
    → May still help robustness to distribution shift

IF aug_test_ppl > baseline_test_ppl:
    → Augmentation may be adding noise
    → Check which augmentation types are problematic
```

## Statistical Considerations

With n=38 test samples:
- Bootstrap 95% CIs are provided for all metrics
- Effect size > 10% is likely significant
- Effect size 5-10% is borderline (may be noise)
- Effect size < 5% is within noise margin

## Directory Structure

```
exp001-augmentation-impact/
├── README.md                 # This file
├── experiment.json           # Experiment configuration
├── split_info.json           # Dataset split metadata
├── train/                    # Raw train split (152 samples)
├── train_formatted/          # Baseline training data (152 samples)
├── train_augmented/          # Augmented training data (436 samples)
├── test/                     # Raw test split (38 samples)
├── test_formatted/           # Test evaluation data (38 samples)
├── models/                   # Trained models
│   ├── baseline/             # 3 epoch checkpoints + final
│   └── augmented/            # 1 epoch checkpoint + final
└── results/                  # Evaluation results
    ├── baseline.json         # Baseline checkpoint evaluation
    ├── augmented.json        # Augmented checkpoint evaluation
    └── comparison.html       # Final comparison report
```

## References

- [ADR-007: Experimental Framework](../../docs/adrs/ADR-007-augmentation-experiment-framework.md)
- [ADR-008: Fair Comparison Methodology](../../docs/adrs/ADR-008-fair-augmentation-comparison-methodology.md)
- [ADR-005: Completion-Only Perplexity](../../docs/adrs/ADR-005-completion-perplexity.md)
- [ADR-006: Chat Template Consistency](../../docs/adrs/ADR-006-chat-template-consistency.md)
