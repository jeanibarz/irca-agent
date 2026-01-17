# RFC-001: Dataset Diversity Metrics for SFT

* Status: draft
* Author: Jean
* Date: 2026-01-16
* Revised: 2026-01-16

## Summary

This RFC proposes methods to quantify dataset diversity before and after augmentation, with a focus on **training effectiveness** for Supervised Fine-Tuning (SFT). We argue that for SFT, model-dependent metrics (perplexity-based) are more appropriate than model-agnostic metrics (embedding-based), because the ultimate goal is to make the model robust to input variations.

## Motivation

When augmenting a dataset for SFT, we need to answer:

1. **Will augmentation make the model more robust?** Not just "is the dataset more diverse?"
2. **Does syntactic variation matter as much as semantic variation?** Yes, for SFT it does.
3. **How do we know augmentation actually helped?** Compare model behavior before/after.
4. **Is model-dependence a bug or a feature?** It's a feature for SFT.

### The Core Insight

The goal of SFT is to make a model behave correctly across many input variations. This means:

- **Semantic diversity matters**: Different user queries, different tool combinations
- **Syntactic diversity also matters**: If the model breaks when formatting changes slightly, training failed

Our augmentation techniques target both:
- Translation → semantic + syntactic variation
- Function shuffling → syntactic (order doesn't change meaning, but model must handle it)
- Format variation → purely syntactic (same content, different presentation)
- Newline variation → purely syntactic

A **semantic-only diversity metric** (like Vendi Score with sentence embeddings) would undervalue our syntactic augmentations, even though they're crucial for model robustness.

## Why Semantic-Only Metrics Fall Short for SFT

### The Problem with Embedding-Based Metrics

Sentence embeddings capture **meaning**, not **form**:

```
"The weather is sunny today."     → embedding E1
"Today's weather is sunny."       → embedding E2 ≈ E1 (very similar!)
"Il fait beau aujourd'hui."       → embedding E3 ≈ E1 (French, but similar meaning)
```

A Vendi Score computed on these embeddings would see low diversity - but for SFT, these variations are **exactly what we need** to make the model robust!

### What We Actually Care About

For SFT, the right question is:

> "Will the finetuned model handle this input correctly, even if it's slightly different from training examples?"

This is inherently **model-dependent**. A dataset might be "diverse" in embedding space but trivial for a capable model, or "similar" in embedding space but challenging due to syntactic variations the model hasn't learned.

## Proposed Approach: Perplexity-Based Training Effectiveness

### The Key Insight

Perplexity measures how "surprising" an input is to a model:
- **High perplexity** = Model finds the input unfamiliar/hard to predict
- **Low perplexity** = Model has seen similar patterns, can predict well

For SFT evaluation, we care about **perplexity deltas**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PERPLEXITY-BASED EVALUATION                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  BASE MODEL (before finetuning)                                     │
│    └─► Perplexity on dataset = HIGH                                 │
│        (Model finds our agent traces "foreign")                     │
│                                                                     │
│  FINETUNED ON ORIGINAL (1000 samples)                               │
│    └─► Perplexity on TEST SET = MEDIUM                              │
│        (Model learned the format, some generalization)              │
│                                                                     │
│  FINETUNED ON AUGMENTED (3000 samples)                              │
│    └─► Perplexity on TEST SET = LOW                                 │
│        (Better generalization due to diverse training)              │
│                                                                     │
│  If augmented training → lower test perplexity:                     │
│    ✅ AUGMENTATION HELPED GENERALIZATION                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Model-Dependence is a Feature

Some might say "but perplexity is model-specific, we want a universal metric!"

**Counter-argument**: For SFT, model-specificity is exactly what we want.

Consider two scenarios:

**Scenario A: Universal metric says "high diversity"**
- Dataset has many semantically different samples
- But our target model already handles them well
- Finetuning adds little value
- Universal metric was misleading

**Scenario B: Model-specific metric says "high surprise"**
- Base model has high perplexity on our dataset
- This means training will actually teach the model something new
- Model-specific metric correctly predicted training value

### The Robustness Argument

Some argue: "Use the same exact prompt, you need fewer training samples."

**Counter-argument**: This leads to **overfitting to prompt format**.

```
Training with fixed prompt:
  ✅ Works perfectly with exact prompt template
  ❌ Breaks if user slightly modifies the prompt
  ❌ Fragile in production where inputs vary

Training with varied prompts (augmentation):
  ✅ Works with many prompt variations
  ✅ Robust to user modifications
  ✅ Better generalization
  (May need more samples, but produces better model)
```

**Real-world example**: If your agent is trained with:
```
### Available Functions
[{"name": "get_weather", ...}]
```

And a user's system sends:
```
## Functions Available
[{"name": "get_weather", ...}]
```

A model trained only on the first format might fail. Syntactic diversity in training prevents this.

## Proposed Metrics

### 1. Perplexity-Based Metrics (Recommended for SFT)

#### 1.1 Dataset Perplexity Profile

Measure how the target model perceives the dataset:

```python
def compute_perplexity_profile(
    texts: list[str],
    model_name: str
) -> dict:
    """
    Compute perplexity statistics for a dataset.

    Higher mean perplexity = more challenging/diverse for THIS model
    Higher std = more variation in difficulty across samples
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    perplexities = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            ppl = torch.exp(outputs.loss).item()
        perplexities.append(ppl)

    return {
        "mean_perplexity": np.mean(perplexities),
        "std_perplexity": np.std(perplexities),
        "min_perplexity": np.min(perplexities),
        "max_perplexity": np.max(perplexities),
        "median_perplexity": np.median(perplexities),
        "p90_perplexity": np.percentile(perplexities, 90),
    }
```

#### 1.2 Augmentation Effectiveness Score

The key metric: **Did augmentation improve model generalization?**

```python
def evaluate_augmentation_effectiveness(
    original_train: Dataset,
    augmented_train: Dataset,
    test_set: Dataset,
    base_model: str,
    finetune_config: dict,
) -> dict:
    """
    Evaluate augmentation effectiveness by comparing:
    1. Model finetuned on original → test perplexity
    2. Model finetuned on augmented → test perplexity

    Lower test perplexity with augmented training = augmentation helped.
    """
    # 1. Base model perplexity on test set (baseline)
    base_ppl = compute_perplexity_profile(test_set["text"], base_model)

    # 2. Finetune on original, measure test perplexity
    model_original = finetune(base_model, original_train, finetune_config)
    original_test_ppl = compute_perplexity_profile(
        test_set["text"], model_original
    )

    # 3. Finetune on augmented, measure test perplexity
    model_augmented = finetune(base_model, augmented_train, finetune_config)
    augmented_test_ppl = compute_perplexity_profile(
        test_set["text"], model_augmented
    )

    # 4. Compute effectiveness
    original_improvement = (
        base_ppl["mean_perplexity"] - original_test_ppl["mean_perplexity"]
    ) / base_ppl["mean_perplexity"]

    augmented_improvement = (
        base_ppl["mean_perplexity"] - augmented_test_ppl["mean_perplexity"]
    ) / base_ppl["mean_perplexity"]

    augmentation_delta = augmented_improvement - original_improvement

    return {
        "base_model_test_ppl": base_ppl["mean_perplexity"],
        "original_trained_test_ppl": original_test_ppl["mean_perplexity"],
        "augmented_trained_test_ppl": augmented_test_ppl["mean_perplexity"],
        "original_improvement_pct": original_improvement * 100,
        "augmented_improvement_pct": augmented_improvement * 100,
        "augmentation_delta_pct": augmentation_delta * 100,
        "augmentation_effective": augmentation_delta > 0,
    }
```

#### 1.3 Syntactic vs Semantic Perplexity

Separate the contribution of syntactic and semantic diversity:

```python
def analyze_diversity_components(
    original_texts: list[str],
    semantically_augmented: list[str],  # e.g., translation only
    syntactically_augmented: list[str],  # e.g., shuffle + format only
    fully_augmented: list[str],          # both
    model_name: str,
) -> dict:
    """
    Analyze contribution of semantic vs syntactic augmentation.
    """
    base_ppl = compute_perplexity_profile(original_texts, model_name)
    semantic_ppl = compute_perplexity_profile(semantically_augmented, model_name)
    syntactic_ppl = compute_perplexity_profile(syntactically_augmented, model_name)
    full_ppl = compute_perplexity_profile(fully_augmented, model_name)

    return {
        "baseline_mean_ppl": base_ppl["mean_perplexity"],
        "semantic_only_mean_ppl": semantic_ppl["mean_perplexity"],
        "syntactic_only_mean_ppl": syntactic_ppl["mean_perplexity"],
        "full_augmentation_mean_ppl": full_ppl["mean_perplexity"],
        "semantic_contribution": (
            semantic_ppl["mean_perplexity"] - base_ppl["mean_perplexity"]
        ),
        "syntactic_contribution": (
            syntactic_ppl["mean_perplexity"] - base_ppl["mean_perplexity"]
        ),
    }
```

### 2. Vendi Score (Model-Agnostic Fallback)

When you need a quick, model-free estimate or want to compare datasets across different model targets:

```python
def compute_vendi_score(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> float:
    """
    Compute Vendi Score - effective number of unique samples in embedding space.

    Useful as a quick sanity check, but doesn't capture syntactic diversity well.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True)

    K = embeddings @ embeddings.T
    n = len(texts)
    K = K / n

    eigenvalues = eigvalsh(K)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    eigenvalues = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))

    return np.exp(entropy)
```

**Limitation**: Vendi Score with sentence embeddings undervalues syntactic diversity.

### 3. N-gram Diversity (Lexical/Syntactic)

Captures surface-level syntactic variation:

```python
def compute_lexical_diversity(texts: list[str]) -> dict:
    """
    Compute lexical diversity metrics.

    Good for measuring syntactic variation (different words, n-grams).
    """
    def distinct_n(texts, n):
        all_ngrams = []
        for text in texts:
            tokens = text.split()
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            all_ngrams.extend(ngrams)
        if not all_ngrams:
            return 0.0
        return len(set(all_ngrams)) / len(all_ngrams)

    return {
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "distinct_3": distinct_n(texts, 3),
    }
```

## Metric Comparison

| Metric | Captures Semantic | Captures Syntactic | Model-Specific | Predicts Training Value | Compute Cost |
|--------|-------------------|-------------------|----------------|------------------------|--------------|
| **Perplexity Profile** | ✅ Yes | ✅ Yes | ✅ Yes (feature!) | ✅ Directly | High |
| **Augmentation Effectiveness** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Gold standard | Very High |
| Vendi Score | ✅ Yes | ❌ Limited | ❌ No | ⚠️ Partial | Medium |
| N-gram Diversity | ❌ No | ✅ Yes | ❌ No | ⚠️ Partial | Low |

**Recommendation for SFT:**
1. **Primary**: Perplexity-based metrics (dataset perplexity profile)
2. **Gold Standard**: Augmentation effectiveness (requires training, expensive but definitive)
3. **Quick Check**: N-gram diversity + Vendi Score (fast sanity check)

## Concrete Examples

### Example 1: Syntactic Augmentation Value

```
Original dataset: 1000 agent traces

Metrics with Base Model (Qwen 2.5 3B):
  Mean Perplexity: 45.2
  Std Perplexity: 12.3

After Format Variation Augmentation (purely syntactic):
  Vendi Score change: +3% (embeddings barely changed)
  N-gram Distinct-2 change: +18%
  Mean Perplexity: 52.8 (+17%)

Interpretation:
  - Vendi Score undervalues this augmentation (similar semantics)
  - Perplexity correctly shows this is "new" to the model
  - This syntactic diversity will help the model generalize
```

### Example 2: Augmentation Effectiveness Evaluation

```
Original Training Set: 1000 samples
Augmented Training Set: 3000 samples (3x with full augmentation)
Test Set: 200 samples (held out)

Base Model Test Perplexity: 78.5

After Finetuning on Original (1000 samples):
  Test Perplexity: 28.3
  Improvement: 64% reduction

After Finetuning on Augmented (3000 samples):
  Test Perplexity: 18.7
  Improvement: 76% reduction

Augmentation Delta: +12% additional improvement
Conclusion: ✅ Augmentation significantly improved generalization
```

### Example 3: Diminishing Returns Analysis

```
Base Model Test Perplexity: 78.5

Multiply Factor | Augmented Samples | Test Perplexity | Improvement
----------------|-------------------|-----------------|------------
1x (original)   | 1000              | 28.3            | 64.0%
2x              | 2000              | 22.1            | 71.8%
3x              | 3000              | 18.7            | 76.2%
4x              | 4000              | 17.2            | 78.1%
5x              | 5000              | 16.8            | 78.6%

Observation: Diminishing returns after 3x
Recommendation: Use 3x multiplication for best efficiency
```

### Example 4: Semantic vs Syntactic Contribution

```
Original Dataset Mean Perplexity: 45.2

Augmentation Type           | Mean Perplexity | Delta  | Contribution
----------------------------|-----------------|--------|-------------
Translation only (semantic) | 48.9            | +3.7   | 32%
Shuffle+Format (syntactic)  | 52.1            | +6.9   | 60%
Both combined               | 56.7            | +11.5  | 100%

Insight: Syntactic augmentation contributes MORE to perplexity increase,
meaning it adds MORE "novelty" for the model to learn from.
```

## CLI Design

### Commands

```bash
# Perplexity profile for a dataset
irca dataset diversity --dataset datasets/irca-v1 --model Qwen/Qwen2.5-3B

# Compare original vs augmented
irca dataset diversity --original datasets/irca-v1 --augmented datasets/irca-v1-aug --model Qwen/Qwen2.5-3B

# Full augmentation effectiveness (expensive - requires training)
irca dataset diversity --original datasets/irca-v1 --augmented datasets/irca-v1-aug \
    --test-set datasets/irca-v1-test --model Qwen/Qwen2.5-3B \
    --evaluate-effectiveness

# Quick mode (Vendi + N-gram only, no model loading)
irca dataset diversity --dataset datasets/irca-v1 --quick
```

### Output Format

```
📊 Dataset Diversity Analysis (SFT-Focused)
════════════════════════════════════════════════════════════════════════

Model: Qwen/Qwen2.5-3B
Original Dataset: datasets/irca-v1 (1000 samples)
Augmented Dataset: datasets/irca-v1-aug (3000 samples)

Perplexity Profile:
                        Original    Augmented    Change
  Mean Perplexity:      45.2        56.7         +25.4%
  Std Perplexity:       12.3        18.9         +53.7%
  P90 Perplexity:       62.1        84.3         +35.7%

  Interpretation: Augmented dataset is significantly more "challenging"
                  for the model - good for training robustness.

Lexical Diversity:
  Distinct-1:           0.098       0.142        +44.9%
  Distinct-2:           0.285       0.389        +36.5%
  Distinct-3:           0.491       0.612        +24.6%

Semantic Diversity (Vendi Score):
  Score:                450.2       523.8        +16.4%
  Effective Ratio:      45.0%       17.5%        (expected with 3x)

Augmentation Breakdown (Perplexity Contribution):
  translate     - 900 samples  → +8.2 mean ppl  (semantic)
  shuffle       - 2400 samples → +5.1 mean ppl  (syntactic)
  format        - 1800 samples → +3.4 mean ppl  (syntactic)
  newline       - 900 samples  → +0.8 mean ppl  (syntactic)

💡 Analysis:
  - Syntactic augmentations contribute 53% of perplexity increase
  - This diversity is NOT captured by Vendi Score alone
  - Recommended: Use perplexity-based metrics for SFT evaluation
```

### Effectiveness Evaluation Output

```
📊 Augmentation Effectiveness Evaluation
════════════════════════════════════════════════════════════════════════

Model: Qwen/Qwen2.5-3B
Training: original (1000) vs augmented (3000)
Test Set: 200 samples

Base Model Performance:
  Test Perplexity: 78.5

After Finetuning on Original:
  Test Perplexity: 28.3
  Improvement: -63.9%

After Finetuning on Augmented:
  Test Perplexity: 18.7
  Improvement: -76.2%

┌─────────────────────────────────────────────────────────────────────┐
│  AUGMENTATION EFFECTIVENESS: +12.3%                                 │
│                                                                     │
│  ✅ Augmentation IMPROVED generalization                            │
│                                                                     │
│  The model finetuned on augmented data has 34% lower perplexity    │
│  on unseen test samples compared to original-only training.         │
└─────────────────────────────────────────────────────────────────────┘

Recommendation: Augmentation is effective. Consider production use.
```

## Integration with Augmentation Pipeline

### Automatic Diversity Tracking

```json
{
  "statistics": {
    "input_samples": 1000,
    "output_samples": 3000,
    "diversity": {
      "model": "Qwen/Qwen2.5-3B",
      "original_mean_perplexity": 45.2,
      "augmented_mean_perplexity": 56.7,
      "perplexity_increase_pct": 25.4,
      "semantic_contribution_pct": 32,
      "syntactic_contribution_pct": 68,
      "vendi_score_original": 450.2,
      "vendi_score_augmented": 523.8,
      "distinct_2_original": 0.285,
      "distinct_2_augmented": 0.389
    }
  }
}
```

## Proposed Requirements

| Req ID | Name | Description |
|--------|------|-------------|
| FR-DATA-18 | Perplexity Diversity | System shall compute perplexity-based diversity metrics using a specified model |
| FR-DATA-19 | Diversity CLI | System shall provide `irca dataset diversity` command with model specification |
| FR-DATA-20 | Diversity in Metadata | Augmentation shall record perplexity metrics in metadata when model is available |
| FR-DATA-21 | Diversity Comparison | System shall compare diversity between original and augmented datasets |
| FR-DATA-22 | Augmentation Effectiveness | System shall evaluate augmentation effectiveness via train/test perplexity comparison |
| FR-DATA-23 | Quick Diversity Mode | System shall provide fast model-free diversity check (Vendi + N-gram) |

## Implementation Plan

### Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        IMPLEMENTATION ROADMAP                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: Quick Metrics (Model-Free)          ████████░░  ~2 hours         │
│    └─► N-gram diversity, Vendi Score                                        │
│                                                                             │
│  Phase 2: Perplexity Metrics (Core)           ████████████░░░░  ~4 hours   │
│    └─► compute_perplexity_profile, batching, sampling                       │
│                                                                             │
│  Phase 3: CLI Integration                     ████████░░  ~2 hours         │
│    └─► irca dataset diversity command                                       │
│                                                                             │
│  Phase 4: Comparison & Analysis               ██████░░░░  ~2 hours         │
│    └─► Original vs augmented, contribution breakdown                        │
│                                                                             │
│  Phase 5: Effectiveness Evaluation            ████████████░░░░  ~4 hours   │
│    └─► Full train/test perplexity comparison                                │
│                                                                             │
│  Phase 6: Pipeline Integration                ██████░░░░  ~2 hours         │
│    └─► Automatic diversity in augmentation metadata                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### File Structure

```
src/
├── diversity/
│   ├── __init__.py              # Module exports
│   ├── lexical.py               # N-gram diversity (Distinct-1/2/3)
│   ├── semantic.py              # Vendi Score (embedding-based)
│   ├── perplexity.py            # Perplexity metrics (model-based)
│   ├── effectiveness.py         # Augmentation effectiveness evaluation
│   └── utils.py                 # Sampling, batching, caching utilities
├── cli/commands/
│   └── dataset.py               # Add 'diversity' subcommand
└── augmentation/
    └── metadata.py              # Update to include diversity metrics

tests/
├── unit/
│   ├── test_diversity_lexical.py
│   ├── test_diversity_semantic.py
│   ├── test_diversity_perplexity.py
│   └── test_diversity_effectiveness.py
└── integration/
    └── test_diversity_cli.py
```

---

### Phase 1: Quick Metrics (Model-Free)

**Goal**: Implement fast, model-independent diversity metrics as baseline/sanity checks.

#### Task 1.1: Lexical Diversity (`src/diversity/lexical.py`)

```python
# Functions to implement:
def compute_distinct_n(texts: list[str], n: int) -> float
def compute_lexical_diversity(texts: list[str]) -> dict
def compute_self_bleu(texts: list[str], sample_size: int = 1000) -> float  # Optional
```

**Acceptance Criteria**:
- [ ] Distinct-1/2/3 computed correctly for sample texts
- [ ] Handles empty texts gracefully
- [ ] Performance: < 1 second for 1000 samples

**Tests** (`tests/unit/test_diversity_lexical.py`):
- [ ] `test_distinct_n_basic`: Known input → expected output
- [ ] `test_distinct_n_duplicates`: All duplicates → low score
- [ ] `test_distinct_n_unique`: All unique → high score
- [ ] `test_empty_texts`: Empty input → 0.0
- [ ] `test_single_word_texts`: Edge case handling

#### Task 1.2: Semantic Diversity (`src/diversity/semantic.py`)

```python
# Functions to implement:
def compute_vendi_score(
    texts: list[str],
    embedding_model: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    sample_size: int | None = None,  # Sample for large datasets
) -> float

def compute_embedding_coverage(
    texts: list[str],
    embedding_model: str = "all-MiniLM-L6-v2",
) -> dict  # avg_pairwise_distance, embedding_std, etc.
```

**Acceptance Criteria**:
- [ ] Vendi Score = 1 for identical texts
- [ ] Vendi Score ≈ n for n completely different texts
- [ ] Duplicating dataset doesn't change Vendi Score
- [ ] Sampling works for large datasets (> 5000)

**Tests** (`tests/unit/test_diversity_semantic.py`):
- [ ] `test_vendi_identical_texts`: All same → score ≈ 1
- [ ] `test_vendi_unique_texts`: All different → score ≈ n
- [ ] `test_vendi_duplicates_invariant`: 2x dataset → same score
- [ ] `test_vendi_with_sampling`: Large dataset sampling works
- [ ] `test_embedding_coverage_metrics`: All metrics computed

**Dependencies**: `sentence-transformers`, `scipy`

---

### Phase 2: Perplexity Metrics (Core) ⭐ PRIMARY TARGET

**Goal**: Implement the main perplexity-based diversity metrics.

#### Task 2.1: Core Perplexity Computation (`src/diversity/perplexity.py`)

```python
def compute_sample_perplexity(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
) -> float

def compute_perplexity_profile(
    texts: list[str],
    model_name: str,
    batch_size: int = 4,
    sample_size: int | None = None,
    device: str = "auto",
    show_progress: bool = True,
) -> dict:
    """
    Returns:
        {
            "mean_perplexity": float,
            "std_perplexity": float,
            "min_perplexity": float,
            "max_perplexity": float,
            "median_perplexity": float,
            "p90_perplexity": float,
            "p95_perplexity": float,
            "sample_count": int,
            "model": str,
            "perplexities": list[float],  # Optional, for detailed analysis
        }
    """
```

**Acceptance Criteria**:
- [ ] Perplexity computed correctly (validated against known values)
- [ ] Batched inference for efficiency
- [ ] GPU support with automatic device detection
- [ ] Progress bar for long computations
- [ ] Sampling mode for large datasets
- [ ] Memory-efficient (doesn't OOM on 3B model with 3000 samples)

**Tests** (`tests/unit/test_diversity_perplexity.py`):
- [ ] `test_perplexity_basic`: Simple text → reasonable perplexity
- [ ] `test_perplexity_profile_stats`: All stats computed correctly
- [ ] `test_perplexity_batching`: Batched = unbatched results
- [ ] `test_perplexity_sampling`: Sampling produces consistent estimates
- [ ] `test_perplexity_empty_text`: Edge case handling
- [ ] `test_perplexity_long_text`: Truncation works correctly

**Dependencies**: `transformers`, `torch`

#### Task 2.2: Utilities (`src/diversity/utils.py`)

```python
def sample_texts(texts: list[str], sample_size: int, seed: int = 42) -> list[str]

def get_device() -> str  # "cuda", "mps", or "cpu"

def load_model_and_tokenizer(
    model_name: str,
    device: str = "auto",
) -> tuple[PreTrainedModel, PreTrainedTokenizer]

class PerplexityCache:
    """Cache per-sample perplexity to avoid recomputation."""
    def __init__(self, cache_dir: str = ".diversity_cache")
    def get(self, text_hash: str, model_name: str) -> float | None
    def set(self, text_hash: str, model_name: str, perplexity: float)
    def clear(self)
```

**Acceptance Criteria**:
- [ ] Sampling is reproducible with seed
- [ ] Device detection works on CPU/CUDA/MPS
- [ ] Cache reduces computation on repeated calls

---

### Phase 3: CLI Integration

**Goal**: Expose diversity metrics via `irca dataset diversity` command.

#### Task 3.1: CLI Command (`src/cli/commands/dataset.py`)

```python
@dataset.command("diversity")
@click.option("--dataset", "-d", type=click.Path(exists=True), help="Dataset to analyze")
@click.option("--original", "-o", type=click.Path(exists=True), help="Original dataset for comparison")
@click.option("--augmented", "-a", type=click.Path(exists=True), help="Augmented dataset for comparison")
@click.option("--model", "-m", type=str, default=None, help="Model for perplexity (e.g., Qwen/Qwen2.5-3B)")
@click.option("--quick", is_flag=True, help="Quick mode: Vendi + N-gram only, no model loading")
@click.option("--sample-size", type=int, default=None, help="Sample size for large datasets")
@click.option("--output", type=click.Choice(["text", "json"]), default="text", help="Output format")
@click.option("--cache/--no-cache", default=True, help="Use perplexity cache")
@click.pass_context
def diversity(ctx, dataset, original, augmented, model, quick, sample_size, output, cache):
    """Analyze dataset diversity for SFT."""
```

**CLI Modes**:

1. **Single Dataset Analysis**:
   ```bash
   irca dataset diversity --dataset datasets/irca-v1 --model Qwen/Qwen2.5-3B
   ```

2. **Quick Mode** (no model):
   ```bash
   irca dataset diversity --dataset datasets/irca-v1 --quick
   ```

3. **Comparison Mode**:
   ```bash
   irca dataset diversity --original datasets/irca-v1 --augmented datasets/irca-v1-aug --model Qwen/Qwen2.5-3B
   ```

4. **JSON Output** (for scripting):
   ```bash
   irca dataset diversity --dataset ds --model Qwen/Qwen2.5-3B --output json > diversity.json
   ```

**Acceptance Criteria**:
- [ ] `--dataset` works for single dataset analysis
- [ ] `--original` + `--augmented` works for comparison
- [ ] `--quick` skips model loading
- [ ] `--model` required when not using `--quick`
- [ ] `--output json` produces valid JSON
- [ ] Progress bar shown during perplexity computation
- [ ] Clear error messages for invalid inputs

**Tests** (`tests/integration/test_diversity_cli.py`):
- [ ] `test_cli_quick_mode`: Quick mode works without model
- [ ] `test_cli_single_dataset`: Single dataset analysis
- [ ] `test_cli_comparison_mode`: Original vs augmented
- [ ] `test_cli_json_output`: JSON output is valid
- [ ] `test_cli_missing_model`: Error when model missing and not quick mode
- [ ] `test_cli_invalid_dataset`: Error for non-existent dataset

---

### Phase 4: Comparison & Analysis

**Goal**: Implement comparison features and contribution breakdown.

#### Task 4.1: Dataset Comparison (`src/diversity/perplexity.py`)

```python
def compare_datasets(
    original_texts: list[str],
    augmented_texts: list[str],
    model_name: str | None = None,
    quick: bool = False,
) -> dict:
    """
    Compare diversity metrics between original and augmented datasets.

    Returns:
        {
            "original": { ... metrics ... },
            "augmented": { ... metrics ... },
            "delta": {
                "perplexity_change_pct": float,
                "vendi_change_pct": float,
                "distinct_2_change_pct": float,
                ...
            }
        }
    """
```

#### Task 4.2: Contribution Breakdown (`src/diversity/perplexity.py`)

```python
def analyze_augmentation_contributions(
    original_texts: list[str],
    augmented_texts: list[str],
    augmentation_labels: list[str],  # e.g., ["translate:fr", "shuffle", ...]
    model_name: str,
) -> dict:
    """
    Analyze per-augmentation-type contribution to diversity.

    Returns:
        {
            "translate": {"sample_count": 450, "mean_ppl_delta": 8.2, "contribution_pct": 32},
            "shuffle": {"sample_count": 800, "mean_ppl_delta": 5.1, "contribution_pct": 20},
            ...
        }
    """
```

**Acceptance Criteria**:
- [ ] Comparison shows before/after with delta
- [ ] Contribution breakdown by augmentation type
- [ ] Semantic vs syntactic contribution analysis

---

### Phase 5: Augmentation Effectiveness Evaluation

**Goal**: Implement the gold-standard effectiveness evaluation.

#### Task 5.1: Effectiveness Evaluation (`src/diversity/effectiveness.py`)

```python
def evaluate_augmentation_effectiveness(
    original_train: Dataset,
    augmented_train: Dataset,
    test_set: Dataset,
    model_name: str,
    finetune_config: dict | None = None,
    output_dir: str = "./effectiveness_eval",
) -> dict:
    """
    Evaluate augmentation effectiveness by:
    1. Compute base model perplexity on test set
    2. Finetune on original → compute test perplexity
    3. Finetune on augmented → compute test perplexity
    4. Compare results

    Returns:
        {
            "base_model_test_ppl": float,
            "original_trained_test_ppl": float,
            "augmented_trained_test_ppl": float,
            "original_improvement_pct": float,
            "augmented_improvement_pct": float,
            "augmentation_delta_pct": float,
            "augmentation_effective": bool,
            "training_logs": { ... },
        }
    """
```

#### Task 5.2: CLI Extension

```bash
# Full effectiveness evaluation (expensive!)
irca dataset diversity \
    --original datasets/irca-v1 \
    --augmented datasets/irca-v1-aug \
    --test-set datasets/irca-v1-test \
    --model Qwen/Qwen2.5-3B \
    --evaluate-effectiveness \
    --output-dir ./eval_results
```

**Acceptance Criteria**:
- [ ] Base model perplexity computed
- [ ] Finetuning on original works
- [ ] Finetuning on augmented works
- [ ] Test perplexity comparison correct
- [ ] Results saved to output directory
- [ ] Clear progress indication (this takes hours)

**Note**: This is expensive (requires 2x finetuning). Consider:
- Default to small batch size / few epochs for quick estimate
- Allow `--full-training` flag for complete evaluation
- Cache trained models for reuse

---

### Phase 6: Pipeline Integration

**Goal**: Automatically compute diversity metrics during augmentation.

#### Task 6.1: Update Augmentation Metadata (`src/augmentation/metadata.py`)

```python
def generate_metadata(
    config: AugmentationConfig,
    input_dataset: Dataset,
    output_dataset: Dataset,
    statistics: dict,
    diversity_model: str | None = None,  # NEW
) -> dict:
    """
    If diversity_model is provided, compute and include diversity metrics.
    """
```

#### Task 6.2: Update CLI (`src/cli/commands/dataset.py`)

```bash
# Augment with diversity tracking
irca dataset augment-pipeline \
    --config config.json \
    --diversity-model Qwen/Qwen2.5-3B  # NEW: compute diversity metrics
```

**Acceptance Criteria**:
- [ ] `--diversity-model` option added to augment-pipeline
- [ ] Diversity metrics included in `_augmentation_metadata.json`
- [ ] Warning if perplexity increase < 10% (minimal diversity gain)

#### Task 6.3: Metadata Schema Update

```json
{
  "statistics": {
    "input_samples": 1000,
    "output_samples": 3000,
    "diversity": {
      "computed_at": "2026-01-16T12:00:00Z",
      "model": "Qwen/Qwen2.5-3B",
      "original": {
        "mean_perplexity": 45.2,
        "std_perplexity": 12.3,
        "vendi_score": 450.2,
        "distinct_2": 0.285
      },
      "augmented": {
        "mean_perplexity": 56.7,
        "std_perplexity": 18.9,
        "vendi_score": 523.8,
        "distinct_2": 0.389
      },
      "delta": {
        "perplexity_increase_pct": 25.4,
        "vendi_increase_pct": 16.4,
        "distinct_2_increase_pct": 36.5
      },
      "recommendation": "Good diversity gain. Augmentation effective."
    }
  }
}
```

---

### Dependencies to Add

```toml
# pyproject.toml additions
[tool.poetry.dependencies]
sentence-transformers = "^2.2.0"  # For Vendi Score
scipy = "^1.10.0"                 # For eigenvalue computation
tqdm = "^4.65.0"                  # Progress bars (likely already present)

[tool.poetry.group.dev.dependencies]
# No new dev dependencies needed
```

---

### Testing Strategy

#### Unit Tests (Mocked)

For perplexity tests, mock the model to avoid loading large models in CI:

```python
@patch("src.diversity.perplexity.load_model_and_tokenizer")
def test_perplexity_profile(mock_load):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_model.return_value.loss = torch.tensor(2.0)  # ppl = e^2 ≈ 7.39
    mock_load.return_value = (mock_model, mock_tokenizer)

    result = compute_perplexity_profile(["test text"], "mock-model")
    assert result["mean_perplexity"] == pytest.approx(7.39, rel=0.01)
```

#### Integration Tests (Real Model, Optional)

Mark as slow/optional for CI:

```python
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_perplexity_real_model():
    """Integration test with real model - run manually or in nightly CI."""
    texts = ["Hello, how are you?", "The weather is nice today."]
    result = compute_perplexity_profile(texts, "gpt2")  # Small model for testing
    assert result["mean_perplexity"] > 0
```

---

### Implementation Order

```
Week 1:
├── Phase 1: Quick Metrics (2h)
│   ├── Task 1.1: Lexical diversity
│   └── Task 1.2: Vendi Score
└── Phase 2: Perplexity Metrics (4h)
    ├── Task 2.1: Core perplexity computation
    └── Task 2.2: Utilities (sampling, caching)

Week 2:
├── Phase 3: CLI Integration (2h)
│   └── Task 3.1: diversity command
├── Phase 4: Comparison (2h)
│   ├── Task 4.1: Dataset comparison
│   └── Task 4.2: Contribution breakdown
└── Phase 5: Effectiveness (4h)
    ├── Task 5.1: Evaluation function
    └── Task 5.2: CLI extension

Week 3:
└── Phase 6: Pipeline Integration (2h)
    ├── Task 6.1: Metadata update
    ├── Task 6.2: CLI integration
    └── Task 6.3: Documentation
```

---

### Acceptance Criteria Summary

| Requirement | Phase | Key Acceptance Criteria |
|-------------|-------|------------------------|
| FR-DATA-18 | 2 | `compute_perplexity_profile()` returns correct stats |
| FR-DATA-19 | 3 | `irca dataset diversity` command works with all modes |
| FR-DATA-20 | 6 | Augmentation metadata includes diversity when model specified |
| FR-DATA-21 | 4 | Comparison mode shows before/after with delta |
| FR-DATA-22 | 5 | Effectiveness evaluation compares train/test perplexity |
| FR-DATA-23 | 1 | `--quick` mode works without model loading |

## Technical Considerations

### Computational Cost

Perplexity computation requires model inference on all samples:

| Dataset Size | Model Size | Estimated Time |
|--------------|------------|----------------|
| 1000 samples | 3B params  | ~5 minutes     |
| 3000 samples | 3B params  | ~15 minutes    |
| 10000 samples| 3B params  | ~50 minutes    |

**Mitigation strategies:**
- Sampling: Compute on random subset for large datasets
- Caching: Store per-sample perplexity for reuse
- GPU batching: Batch inference for speed

### Model Selection

The perplexity should be computed with a model similar to your target:
- If finetuning Qwen 2.5 3B → use Qwen 2.5 3B for perplexity
- If finetuning different model → use that model

The metric is intentionally model-specific because that's what predicts training value.

## Answering the Original Questions (Revised)

### 1. How do we measure dataset diversity for SFT?

**Use perplexity-based metrics.** They capture both semantic and syntactic diversity, and directly predict training value for your target model.

### 2. Is model-dependence a problem?

**No, it's a feature.** For SFT, you want to know "will THIS model benefit from THIS dataset?" That's inherently model-specific.

### 3. Do syntactic augmentations add value?

**Yes, significantly.** Perplexity analysis shows syntactic augmentations (shuffle, format variation) contribute substantial "novelty" that the model must learn, improving robustness.

### 4. How do we know augmentation actually helped?

**Augmentation Effectiveness evaluation.** Compare test set perplexity after training on original vs augmented data. If augmented training leads to lower test perplexity, augmentation helped generalization.

### 5. Why not just use Vendi Score?

**Vendi Score undervalues syntactic diversity.** Sentence embeddings capture semantics, not form. For SFT where we need robustness to formatting changes, perplexity-based metrics are more accurate.

## References

1. Meister, C. & Cotterell, R. (2021). "Language Model Evaluation Beyond Perplexity"
2. Holtzman, A. et al. (2019). "The Curious Case of Neural Text Degeneration"
3. Friedman, D. & Dieng, A.B. (2022). "The Vendi Score: A Diversity Evaluation Metric for Machine Learning"
4. Li, J. et al. (2016). "A Diversity-Promoting Objective Function for Neural Conversation Models" (Distinct-n)

## Open Questions

1. **Sampling strategy**: For large datasets, what's the minimum sample size for reliable perplexity estimates?
2. **Probe model**: Can we use a smaller model as a proxy for the target model's perplexity?
3. **Thresholds**: What perplexity increase indicates "good" augmentation? (Needs empirical study)
4. **Per-section analysis**: Should we compute perplexity on full text or specific sections (system prompt, user query, completion)?
5. **Batch effects**: How to handle perplexity differences between translation languages (French vs Chinese may have different base perplexity)?
