# ADR-007: Experimental Framework for Augmentation Impact Analysis

* Status: proposed
* Deciders: Jean
* Date: 2026-01-17

Technical Story: Design a rigorous experimental framework to measure the true impact of data augmentation on model training, generalization, and robustness.

## Context and Problem Statement

We have implemented various augmentation strategies (translation, function shuffling, newline variation) but lack a rigorous framework to measure their actual impact. Current workflow:

```
Original Dataset → Augment → Format → Finetune → Evaluate
```

**Key Questions We Cannot Currently Answer:**
1. Does augmentation improve generalization, or just inflate training metrics?
2. Does training on augmented data make the model more robust to linguistic variation?
3. Is augmentation causing the model to memorize surface patterns instead of learning task structure?
4. Which augmentation strategies provide actual value vs. adding noise?

### Current Dataset State

| Dataset | Samples | Description |
|---------|---------|-------------|
| `irca_agent_dataset_v5-5acc` | 190 | Original annotated traces |
| `irca_agent_dataset_v5-5acc-augmented` | 190 | With text column |
| `irca_agent_dataset_v5-5acc-formatted-augmented` | 554 | 3x multiplied with augmentations |

**Problem:** With no train/test split, we cannot distinguish overfitting from genuine learning.

## Critical Assessment: Is This Useful?

### Arguments FOR This Approach

1. **Scientific Rigor**: This is standard ML practice. Without controlled experiments, any "improvement" claims are unfounded.

2. **Augmentation Validation**: Many augmentation strategies are borrowed from NLP literature without validation on agentic tasks. Translation augmentation may help for user-facing text but could harm reasoning traces.

3. **Resource Optimization**: If certain augmentations don't help (or hurt), we can remove them and reduce training time.

4. **Debugging Capability**: When a model fails in production, this framework helps identify if the failure is due to:
   - Training data quality
   - Augmentation artifacts
   - Genuine OOD samples

5. **Publication Quality**: If you ever want to publish findings about IRCA training, rigorous experimental methodology is mandatory.

### Arguments AGAINST / Concerns

1. **Dataset Size Problem**:
   - 190 samples is VERY small for ML standards
   - 80/20 split gives ~150 train, ~40 test
   - Results will have high variance, may not be statistically significant
   - **Mitigation**: Use k-fold cross-validation instead of single split

2. **Computational Cost**:
   - 4 dataset variants × potentially multiple augmentation configs = many datasets
   - 2+ model variants per experiment
   - QLoRA finetuning still takes ~1-2 hours per run
   - **Mitigation**: Focus on key comparisons, not full factorial

3. **Augmentation Leakage Risk**:
   - If original sample A is in train, and augmented(A) is semantically similar
   - Train/test split on originals BEFORE augmentation is critical
   - **Mitigation**: Always split originals first, then augment each split independently

4. **Overfitting to Evaluation**:
   - Risk of tuning augmentations to maximize test metrics
   - Could need a held-out "production" set never used for tuning
   - **Mitigation**: Reserve 10% as truly held-out validation

5. **Complexity Debt**:
   - More infrastructure to maintain
   - Higher barrier to entry for contributors
   - **Mitigation**: Good automation and clear documentation

### My Honest Opinion

**This IS useful, but with caveats:**

The approach is scientifically sound and would provide genuine insights. However, with only 190 samples:

1. **Use k-fold cross-validation** (5-fold) instead of single split to get more robust estimates
2. **Focus on binary comparisons** initially: augmented vs. non-augmented training
3. **Measure effect sizes**, not just whether one is "better"
4. **Accept that results may be noisy** - this is informative too (means you need more data)

If the experiments show augmentation provides < 5% improvement with high variance, that's valuable information - it means you should invest in MORE data collection, not more augmentation tricks.

## Proposed Experimental Framework

### 1. Dataset Splitting Strategy

```
Original Dataset (190 samples)
├── Train Pool (80%, ~152 samples)
│   ├── Train-Original (no augmentation, just format)
│   └── Train-Augmented (augmented, then format)
├── Test Pool (20%, ~38 samples)
│   ├── Test-Original (no augmentation, just format)
│   └── Test-Augmented (augmented, then format)
└── [Optional] Holdout (10%, ~19 samples)
    └── Never used during experimentation
```

**Critical Rule:** Split happens on ORIGINAL data BEFORE any augmentation.

### 2. Experimental Matrix

| Experiment | Training Data | Test Data | Measures |
|------------|--------------|-----------|----------|
| E1: Baseline | Train-Original | Test-Original | Base performance |
| E2: Aug-Train | Train-Augmented | Test-Original | Does aug help generalization? |
| E3: Aug-Test | Train-Original | Test-Augmented | Robustness to variations? |
| E4: Full-Aug | Train-Augmented | Test-Augmented | Combined effect |

**Key Comparisons:**
- E2 vs E1: Augmentation training benefit
- E3 vs E1: Model robustness to variations
- E4 vs E2: Does aug-test "cheat" by matching aug-train distribution?

### 3. Metrics to Collect

| Metric | Description | Why |
|--------|-------------|-----|
| Completion Perplexity (mean) | Average PPL on assistant response | Primary quality metric |
| Completion Perplexity (std) | Variance across samples | Consistency measure |
| Generalization Gap | (Test PPL - Train PPL) / Train PPL | Overfitting indicator |
| Per-Language PPL | PPL broken down by query language | Translation effectiveness |
| Per-Augmentation PPL | PPL on specific augmentation types | Which augmentations help? |

### 4. Statistical Considerations

With n=38 test samples:
- Use **paired comparisons** where possible (same original evaluated by both models)
- Report **confidence intervals**, not just point estimates
- Use **bootstrap** for variance estimation
- Consider **Wilcoxon signed-rank test** for significance (non-parametric, small n)

## Proposed Architecture

### New CLI Commands

```bash
# 1. Split original dataset into train/test
irca dataset split \
  --input datasets/irca_agent_dataset_v5-5acc \
  --output-dir experiments/exp001 \
  --train-ratio 0.8 \
  --seed 42

# Creates:
#   experiments/exp001/train_original/
#   experiments/exp001/test_original/
#   experiments/exp001/split_info.json

# 2. Create augmented versions of each split
irca dataset format \
  --input experiments/exp001/train_original \
  --output experiments/exp001/train_augmented \
  --config configs/augment_full.json

irca dataset format \
  --input experiments/exp001/test_original \
  --output experiments/exp001/test_augmented \
  --config configs/augment_full.json

# Also create non-augmented formatted versions
irca dataset format \
  --input experiments/exp001/train_original \
  --output experiments/exp001/train_formatted \
  --no-augment

# 3. Run experiment batch
irca experiment run \
  --config experiments/exp001/experiment.json \
  --output experiments/exp001/results/

# 4. Generate comparison report
irca experiment report \
  --input experiments/exp001/results/ \
  --output experiments/exp001/report.html
```

### Experiment Configuration Schema

```json
{
  "experiment_id": "exp001-augmentation-impact",
  "description": "Measure impact of translation augmentation on model robustness",
  "seed": 42,

  "datasets": {
    "train_original": "experiments/exp001/train_formatted",
    "train_augmented": "experiments/exp001/train_augmented",
    "test_original": "experiments/exp001/test_formatted",
    "test_augmented": "experiments/exp001/test_augmented"
  },

  "models": {
    "baseline": {
      "train_data": "train_original",
      "base_model": "Qwen/Qwen2.5-3B-Instruct",
      "output_dir": "experiments/exp001/models/baseline"
    },
    "augmented": {
      "train_data": "train_augmented",
      "base_model": "Qwen/Qwen2.5-3B-Instruct",
      "output_dir": "experiments/exp001/models/augmented"
    }
  },

  "evaluations": [
    {"model": "baseline", "test_data": "test_original", "name": "E1"},
    {"model": "augmented", "test_data": "test_original", "name": "E2"},
    {"model": "baseline", "test_data": "test_augmented", "name": "E3"},
    {"model": "augmented", "test_data": "test_augmented", "name": "E4"}
  ],

  "metrics": ["completion_perplexity", "generalization_gap", "per_language_breakdown"],
  "statistical_tests": ["paired_bootstrap", "wilcoxon"]
}
```

### Directory Structure

```
experiments/
├── exp001-augmentation-impact/
│   ├── experiment.json          # Experiment configuration
│   ├── split_info.json          # Dataset split metadata
│   ├── train_original/          # Split: train (raw)
│   ├── train_formatted/         # Split: train (formatted, no aug)
│   ├── train_augmented/         # Split: train (formatted + augmented)
│   ├── test_original/           # Split: test (raw)
│   ├── test_formatted/          # Split: test (formatted, no aug)
│   ├── test_augmented/          # Split: test (formatted + augmented)
│   ├── models/
│   │   ├── baseline/            # Model trained on train_formatted
│   │   └── augmented/           # Model trained on train_augmented
│   ├── results/
│   │   ├── E1_baseline_test-original.json
│   │   ├── E2_augmented_test-original.json
│   │   ├── E3_baseline_test-augmented.json
│   │   └── E4_augmented_test-augmented.json
│   └── report.html              # Comparative analysis
```

## Implementation Phases

### Phase 1: Dataset Splitting (Low effort)
- Add `irca dataset split` command
- Implement stratified splitting (if metadata allows)
- Generate split metadata for reproducibility

### Phase 2: Experiment Configuration (Medium effort)
- Define experiment JSON schema
- Add `irca experiment run` orchestrator
- Implement batch training with consistent hyperparameters

### Phase 3: Comparative Evaluation (Medium effort)
- Extend diversity evaluation for paired comparisons
- Add statistical significance testing
- Implement per-augmentation breakdown

### Phase 4: Reporting (Medium effort)
- Add `irca experiment report` command
- Generate HTML report with comparison tables
- Include statistical significance indicators
- Visualizations: PPL distributions, language breakdowns

## Alternatives Considered

### Alternative 1: Just Use Cross-Validation
Instead of single train/test split, use 5-fold CV for all experiments.

**Pros:** More robust estimates, uses all data for both training and testing
**Cons:** 5x training cost, more complex orchestration
**Decision:** Could be added as option in Phase 2

### Alternative 2: Bootstrap from Training Data
Use bootstrap sampling to create synthetic test sets.

**Pros:** No data "wasted" on held-out test
**Cons:** Test distribution matches train distribution exactly, doesn't test generalization
**Decision:** Not recommended for this use case

### Alternative 3: External Test Set
Use a completely different dataset for testing (e.g., different task domain).

**Pros:** True OOD generalization test
**Cons:** May not be available, apples-to-oranges comparison
**Decision:** Good addition if available, not a replacement

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Results too noisy due to small n | High | Medium | Use bootstrap CI, report effect sizes |
| Augmentation leakage across splits | Medium | High | Always split before augment, verify |
| Over-tuning to test set | Medium | Medium | Reserve holdout set, pre-register experiments |
| Computational cost | Medium | Low | Focus on key comparisons, use smaller model |

## Decision

**Proceed with Phase 1 and Phase 2** to enable basic experimentation. This provides immediate value with reasonable investment.

**Defer Phase 3 and Phase 4** until initial results indicate whether deeper analysis is warranted.

## Success Criteria

The framework is successful if:
1. We can answer "Does augmentation improve test performance?" with confidence intervals
2. Results are reproducible (same seeds → same results)
3. Time to run a new experiment is < 4 hours (2 training runs + evaluation)
4. Report clearly shows which augmentations help vs. hurt

## References

- ADR-005: Completion-Only Perplexity Evaluation
- ADR-006: Chat Template Consistency
- [Data Augmentation for Low Resource Neural Machine Translation](https://arxiv.org/abs/1705.00440)
- [A Survey on Data Augmentation for Text Classification](https://arxiv.org/abs/2107.03158)
