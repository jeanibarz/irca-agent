# ADR-008: Fair Comparison Methodology for Augmentation Experiments

* Status: accepted
* Deciders: Jean
* Date: 2026-01-17

Technical Story: Define a rigorous methodology for comparing models trained on augmented vs. non-augmented data to draw valid conclusions about augmentation effectiveness.

## Context and Problem Statement

When comparing a model trained on augmented data vs. non-augmented data, we face a fundamental experimental design challenge:

- **Non-augmented dataset:** 152 samples
- **Augmented dataset:** 436 samples (2.87x larger)

If we naively train both for "3 epochs," the augmented model sees 3x more gradient updates, making comparison unfair. If we adjust epochs to match steps, the non-augmented model sees the same samples repeatedly, risking overfitting.

**Core Question:** How do we make a fair comparison to determine if augmentation actually helps?

## Decision Drivers

1. **Scientific validity:** Results must support valid conclusions
2. **Practical relevance:** Answer the question "Should I augment my data?"
3. **Diagnostic capability:** Detect overfitting vs. genuine improvement
4. **Reproducibility:** Clear protocol others can follow

## Considered Options

### Option A: Same Number of Epochs

| Dataset | Epochs | Steps (batch=4) | Sample Exposures |
|---------|--------|-----------------|------------------|
| Non-augmented (152) | 3 | 114 | 456 |
| Augmented (436) | 3 | 327 | 1308 |

**Pros:**
- Simple to implement
- Augmented sees more data - that's the point of augmentation
- If augmented overfits less despite more training, that's informative

**Cons:**
- Not controlling for compute budget
- Hard to attribute improvement to augmentation vs. more training

### Option B: Same Number of Steps (RECOMMENDED)

| Dataset | Epochs | Steps (batch=4) | Sample Exposures |
|---------|--------|-----------------|------------------|
| Non-augmented (152) | ~3 | 114 | 456 |
| Augmented (436) | ~1 | 109 | 436 |

**Pros:**
- Controls for compute budget
- Fair comparison of "what can I achieve with N gradient updates?"
- Isolates augmentation effect from training duration effect

**Cons:**
- Non-augmented sees same samples multiple times (overfitting risk)
- But: this overfitting IS the problem augmentation solves

### Option C: Train to Convergence with Early Stopping

Train both until validation loss stops improving.

**Pros:**
- Both reach their "best" performance
- Most rigorous for "which strategy is better?"

**Cons:**
- Requires validation split (further reduces already small dataset)
- More complex to implement
- Different total compute for each

## Decision

**Option B: Same Number of Steps** with diagnostic metrics to detect overfitting.

### Rationale

1. **Controls the right variable:** We want to know "given the same training budget, does augmentation help?"

2. **Overfitting is diagnostic, not confounding:** If non-augmented overfits at step N but augmented doesn't, that IS evidence augmentation helps. We report both train and test metrics to reveal this.

3. **Practical interpretation:** "I have X GPU hours. Should I spend them on augmented or non-augmented data?"

## Methodology

### 1. Training Protocol

```bash
# Calculate steps to match
# Non-augmented: 152 samples / batch_size 4 = 38 steps/epoch
# Augmented: 436 samples / batch_size 4 = 109 steps/epoch
#
# To match ~114 steps:
# - Non-augmented: 3 epochs (114 steps)
# - Augmented: 1 epoch (109 steps) ≈ close enough

# Common hyperparameters
BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
BATCH_SIZE=4
LEARNING_RATE=2e-4
LORA_R=64
SEED=42

# Non-augmented: 3 epochs ≈ 114 steps
irca finetune run \
  --model $BASE_MODEL \
  --dataset experiments/exp001-augmentation-impact/train_formatted \
  --output experiments/exp001-augmentation-impact/models/baseline \
  --epochs 3 \
  --batch-size $BATCH_SIZE \
  --learning-rate $LEARNING_RATE \
  --lora-r $LORA_R \
  --seed $SEED \
  --save-steps 38  # Save every epoch for analysis

# Augmented: 1 epoch ≈ 109 steps
irca finetune run \
  --model $BASE_MODEL \
  --dataset experiments/exp001-augmentation-impact/train_augmented \
  --output experiments/exp001-augmentation-impact/models/augmented \
  --epochs 1 \
  --batch-size $BATCH_SIZE \
  --learning-rate $LEARNING_RATE \
  --lora-r $LORA_R \
  --seed $SEED \
  --save-steps 38  # Save at similar intervals for comparison
```

### 2. Checkpoint Analysis

Save intermediate checkpoints to analyze learning dynamics:

| Checkpoint | Non-Aug Steps | Aug Steps | Analysis |
|------------|---------------|-----------|----------|
| 1 | 38 (epoch 1) | 38 | Early training |
| 2 | 76 (epoch 2) | 76 | Mid training |
| 3 | 114 (epoch 3) | 109 (final) | Final comparison |

For each checkpoint, measure:
- **Train perplexity:** How well does it fit training data?
- **Test perplexity:** How well does it generalize?
- **Generalization gap:** `(test_ppl - train_ppl) / train_ppl`

### 3. Metrics to Report

| Metric | Purpose |
|--------|---------|
| Test Completion PPL | Primary performance metric |
| Train Completion PPL | Detect overfitting |
| Generalization Gap | Quantify overfitting severity |
| PPL by Checkpoint | Learning curve analysis |

### 4. Interpretation Guide

```
IF non_aug_test_ppl < aug_test_ppl:
    → Augmentation hurts or doesn't help
    → Consider: Is augmentation adding noise?

IF aug_test_ppl < non_aug_test_ppl:
    IF non_aug_train_ppl << non_aug_test_ppl (large gap):
        → Non-augmented is overfitting
        → Augmentation helps by preventing overfitting
    ELSE:
        → Augmentation genuinely improves learning
        → More diverse training signal helps

IF non_aug_test_ppl ≈ aug_test_ppl (within 5%):
    → No significant difference
    → Augmentation may not be worth the preprocessing cost
    → Consider: Collect more real data instead
```

### 5. Statistical Considerations

With only 38 test samples:
- Report 95% confidence intervals via bootstrap
- Effect size matters more than p-values
- A 10% difference is likely real; 5% may be noise

```python
# Bootstrap confidence interval
import numpy as np

def bootstrap_ci(perplexities, n_bootstrap=1000, ci=0.95):
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(perplexities, size=len(perplexities), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1-ci)/2 * 100)
    upper = np.percentile(means, (1+ci)/2 * 100)
    return lower, upper
```

## Expected Outcomes

| Outcome | Interpretation | Next Action |
|---------|---------------|-------------|
| Aug significantly better (>10%) | Augmentation works | Keep augmentation, possibly expand |
| Aug slightly better (5-10%) | Marginal benefit | Keep if cheap, otherwise collect data |
| No difference (<5%) | Augmentation neutral | Collect more real data |
| Aug worse | Augmentation hurts | Investigate which augmentations harm |

## Consequences

### Positive
- Clear methodology for valid conclusions
- Diagnostic capability to understand WHY results differ
- Reproducible protocol for future experiments

### Negative
- More complex than naive "train both and compare"
- Requires checkpoint saving and multiple evaluations
- Still limited by small test set size

### Risks
- Small test set means high variance in results
- Mitigation: Report confidence intervals, focus on effect size

## Implementation Checklist

- [x] Split dataset (train/test)
- [x] Create formatted datasets (augmented and non-augmented)
- [ ] Train baseline model (3 epochs, save checkpoints)
- [ ] Train augmented model (1 epoch, save checkpoints)
- [ ] Evaluate all checkpoints on train and test sets
- [ ] Generate comparison report with confidence intervals
- [ ] Draw conclusions based on interpretation guide

## References

- ADR-007: Experimental Framework for Augmentation Impact Analysis
- ADR-005: Completion-Only Perplexity Evaluation
- [A Survey on Data Augmentation for Text Classification](https://arxiv.org/abs/2107.03158)
- [Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/abs/1611.03530)
