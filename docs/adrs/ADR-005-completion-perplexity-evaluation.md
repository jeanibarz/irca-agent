# ADR-005: Completion-Only Perplexity Evaluation for Model Robustness

* Status: proposed
* Deciders: Jean
* Date: 2026-01-16

Technical Story: Enable proper evaluation of finetuned model generalization by measuring perplexity on completions only, with baseline model comparison support.

## Context and Problem Statement

After finetuning a model on augmented data, we need to evaluate whether:
1. The finetuning improved task performance
2. The augmentation improved generalization to unseen/diverse data
3. The model is overfitting to training data

### Current Limitations

The existing diversity evaluation (`src/diversity/perplexity.py`) computes perplexity on **full samples** (prompt + completion). This is problematic for SFT evaluation:

1. **Conflates prompt understanding with generation quality**: We care about completion quality, not prompt prediction
2. **No model comparison**: Cannot compare finetuned vs baseline models on same data
3. **No train/test split awareness**: Cannot detect overfitting
4. **Full-sample bias**: Long prompts dominate the perplexity score

### What We Actually Want to Measure

For SFT evaluation, we care about:
- **Completion perplexity**: How well does the model predict the assistant's response given the context?
- **Generalization gap**: How much worse is the model on held-out test data vs training data?
- **Robustness**: How does the model perform on diverse/OOD test samples?

### IRCA Dataset Format

The formatted datasets use section markers:

```
### INSTRUCTIONS
{system_instructions}

### USER QUERY
{user_query}

### ITERATIVE RESOLUTION CYCLE
{completion}  ← WE WANT TO EVALUATE THIS PART

### FINAL ANSWER
{final_answer}
```

## Decision Drivers

* **Accurate SFT evaluation**: Measure what matters - completion quality
* **Model comparison**: Compare finetuned vs baseline on same data
* **Generalization assessment**: Detect overfitting via train/test gap
* **Backward compatibility**: Don't break existing full-sample mode
* **CLI simplicity**: Easy to use for common evaluation workflows

## Considered Options

### Option 1: Completion-Only Mode (Basic)

Add `--eval-mode completion` flag to evaluate only the completion part.

```bash
irca dataset diversity \
  -d test_set.jsonl \
  --model /path/to/finetuned \
  --eval-mode completion
```

| Pros | Cons |
|------|------|
| Simple implementation | No baseline comparison |
| Backward compatible | Single model only |
| Clear semantics | |

### Option 2: Model Comparison Mode (Recommended)

Add comparison between finetuned and baseline models, with completion-only evaluation.

```bash
irca dataset diversity \
  -d test_set.jsonl \
  --model /path/to/finetuned \
  --baseline-model meta-llama/Llama-3.2-1B \
  --eval-mode completion \
  --report robustness_report.html
```

| Pros | Cons |
|------|------|
| Full evaluation matrix | More complex implementation |
| Measures actual improvement | Requires loading 2 models |
| Detects overfitting | Higher memory usage |
| Publication-ready metrics | |

### Option 3: Comprehensive Robustness Suite

Add full robustness evaluation with multiple test sets and metrics.

```bash
irca model evaluate \
  --model /path/to/finetuned \
  --baseline meta-llama/Llama-3.2-1B \
  --train-set train.jsonl \
  --test-set test.jsonl \
  --ood-set ood_test.jsonl
```

| Pros | Cons |
|------|------|
| Complete evaluation | Major new command |
| All robustness metrics | High complexity |
| Standardized workflow | Scope creep |

## Decision Outcome

**Chosen option: Option 2 - Model Comparison Mode**

This provides the essential capability (completion-only perplexity with baseline comparison) while keeping complexity manageable. It integrates with the existing `diversity` command and HTML report infrastructure.

## Detailed Design

### CLI Interface

```bash
# Basic completion-only evaluation
irca dataset diversity \
  -d datasets/test_set \
  --model /path/to/finetuned-model \
  --eval-mode completion

# With baseline comparison (recommended)
irca dataset diversity \
  -d datasets/test_set \
  --model /path/to/finetuned-model \
  --baseline-model meta-llama/Llama-3.2-1B \
  --eval-mode completion \
  --report robustness_report.html

# Full evaluation matrix (train vs test, finetuned vs baseline)
irca dataset diversity \
  --train-set datasets/train \
  --test-set datasets/test \
  --model /path/to/finetuned-model \
  --baseline-model meta-llama/Llama-3.2-1B \
  --eval-mode completion \
  --report full_evaluation.html
```

### Evaluation Matrix Output

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Model Robustness Evaluation (Completion Perplexity)                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Evaluation Mode: completion-only                                       │
│  Finetuned Model: /path/to/finetuned-model                             │
│  Baseline Model:  meta-llama/Llama-3.2-1B                              │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │              │  Train Set    │  Test Set     │  Gap           │    │
│  ├──────────────┼───────────────┼───────────────┼────────────────┤    │
│  │ Baseline     │  PPL: 45.2    │  PPL: 48.1    │  +6.4%         │    │
│  │ Finetuned    │  PPL: 12.3    │  PPL: 15.8    │  +28.5%        │    │
│  ├──────────────┼───────────────┼───────────────┼────────────────┤    │
│  │ Improvement  │  -72.8%       │  -67.2%       │               │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Interpretation:                                                        │
│  ✅ Finetuning improved completion perplexity by 67-73%                │
│  ⚠️  Train-test gap increased (28.5% vs 6.4%) - some overfitting       │
│  ✅ Still strong improvement on held-out test data                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Implementation Architecture

```
src/diversity/
├── __init__.py
├── perplexity.py         # MODIFY: Add completion-only computation
├── evaluation.py         # NEW: Model comparison evaluation
├── report.py             # MODIFY: Add robustness report section
└── utils.py              # MODIFY: Add completion extraction
```

#### New Functions in `src/diversity/perplexity.py`

```python
def compute_completion_perplexity(
    text: str,
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    completion_marker: str = "### ITERATIVE RESOLUTION CYCLE",
    max_length: int | None = None,
) -> float:
    """
    Compute perplexity for the completion part only.

    The full text is tokenized to provide context, but loss is
    computed only on the completion tokens.

    Args:
        text: Full text including prompt and completion.
        model: The language model.
        tokenizer: The tokenizer.
        completion_marker: Marker that separates prompt from completion.
        max_length: Maximum sequence length.

    Returns:
        Perplexity computed on completion tokens only.
    """
    import torch

    # Find completion start
    marker_pos = text.find(completion_marker)
    if marker_pos == -1:
        # Fallback to full perplexity if no marker
        return compute_sample_perplexity(text, model, tokenizer, max_length)

    prompt = text[:marker_pos + len(completion_marker)]
    completion = text[marker_pos + len(completion_marker):]

    # Tokenize separately to find boundary
    prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    prompt_len = prompt_tokens["input_ids"].shape[1]

    # Tokenize full text
    if max_length is None:
        max_length = getattr(model.config, "max_position_embeddings", 2048)

    full_tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    device = next(model.parameters()).device
    input_ids = full_tokens["input_ids"].to(device)

    # Create labels: -100 for prompt tokens (ignored in loss), actual ids for completion
    labels = input_ids.clone()
    labels[0, :prompt_len] = -100  # Mask prompt tokens

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

    return torch.exp(loss).item()
```

#### New Module: `src/diversity/evaluation.py`

```python
"""Model robustness evaluation for SFT."""

from __future__ import annotations

import logging
from typing import Any

from src.diversity.perplexity import (
    compute_completion_perplexity,
    compute_perplexity_profile,
    compute_sample_perplexity,
)
from src.diversity.utils import (
    format_delta,
    get_device,
    load_model_and_tokenizer,
    sample_texts,
)

logger = logging.getLogger(__name__)


def evaluate_model_robustness(
    texts: list[str],
    finetuned_model: str,
    baseline_model: str | None = None,
    eval_mode: str = "completion",  # "completion" or "full"
    completion_marker: str = "### ITERATIVE RESOLUTION CYCLE",
    sample_size: int | None = None,
    seed: int = 42,
    device: str = "auto",
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    Evaluate model robustness on a dataset.

    Args:
        texts: List of formatted text samples.
        finetuned_model: Path to finetuned model.
        baseline_model: Path to baseline model (optional).
        eval_mode: "completion" for completion-only, "full" for full sample.
        completion_marker: Marker separating prompt from completion.
        sample_size: Number of samples to evaluate.
        seed: Random seed.
        device: Device to use.
        show_progress: Whether to show progress.

    Returns:
        Dictionary with evaluation results.
    """
    ...


def evaluate_generalization(
    train_texts: list[str],
    test_texts: list[str],
    finetuned_model: str,
    baseline_model: str | None = None,
    eval_mode: str = "completion",
    **kwargs,
) -> dict[str, Any]:
    """
    Evaluate generalization by comparing train vs test performance.

    Returns metrics including:
    - Train/test perplexity for each model
    - Generalization gap (train-test difference)
    - Improvement over baseline
    """
    ...
```

### Completion Extraction Logic

The IRCA format uses these markers:

```python
COMPLETION_MARKERS = [
    "### ITERATIVE RESOLUTION CYCLE",  # Primary marker
    "### ASSISTANT",                    # Alternative
    "### RESPONSE",                     # Alternative
]

def extract_completion(text: str) -> tuple[str, str]:
    """
    Extract prompt and completion from formatted text.

    Returns:
        Tuple of (prompt, completion).
    """
    for marker in COMPLETION_MARKERS:
        if marker in text:
            idx = text.find(marker)
            prompt = text[:idx + len(marker)]
            completion = text[idx + len(marker):].strip()
            return prompt, completion

    # Fallback: treat entire text as completion
    return "", text
```

### HTML Report Extension

Add new section to robustness report:

```
┌─────────────── MODEL COMPARISON ────────────────┐
│                                                  │
│  [Grouped Bar Chart: Perplexity by Model]        │
│                                                  │
│         Baseline    Finetuned                    │
│  Train:   ████       ██                          │
│  Test:    ████       ███                         │
│                                                  │
│  [Line Chart: Perplexity Distribution]           │
│                                                  │
│       Baseline (blue) vs Finetuned (green)       │
│   ^                                              │
│   │    ╭──╮                                      │
│   │   ╱    ╲    ╭──╮                             │
│   │  ╱      ╲  ╱    ╲                            │
│   └─────────────────────→                        │
│     10    50   100   150  (perplexity)           │
│                                                  │
│  Key Metrics:                                    │
│  • Finetuning improvement: -67.2% on test        │
│  • Generalization gap: +28.5% (finetuned)        │
│  • Baseline gap: +6.4%                           │
│                                                  │
└──────────────────────────────────────────────────┘
```

## Consequences

### Positive

* **Accurate evaluation**: Measures completion quality, not prompt prediction
* **Detects overfitting**: Train/test gap reveals generalization issues
* **Quantifies improvement**: Shows actual benefit of finetuning
* **Guides augmentation**: Reveals which augmentations help generalization
* **Publication-ready**: Standard metrics for ML papers

### Negative

* **Memory usage**: Loading 2 models requires ~2x GPU memory
* **Computation time**: Evaluating 2 models takes ~2x time
* **Complexity**: More CLI options and code paths

### Neutral

* **Backward compatible**: Existing `--model` flag still works
* **Optional baseline**: Comparison is opt-in

## Requirements

| Req ID | Name | Summary |
|--------|------|---------|
| FR-EVAL-01 | Completion-Only Perplexity | Compute perplexity on completion tokens only |
| FR-EVAL-02 | Baseline Model Comparison | Compare finetuned vs baseline model |
| FR-EVAL-03 | Generalization Gap | Report train vs test perplexity gap |
| FR-EVAL-04 | Evaluation Mode Flag | CLI flag to select full vs completion mode |
| FR-EVAL-05 | Robustness Report | HTML report with model comparison visualizations |
| NFR-EVAL-01 | Memory Efficiency | Support sequential model loading for low-memory systems |

## Acceptance Tests

### FR-EVAL-01: Completion-Only Perplexity
```gherkin
Scenario: Compute completion-only perplexity
  Given a formatted dataset with "### ITERATIVE RESOLUTION CYCLE" markers
  When I compute perplexity with --eval-mode completion
  Then only completion tokens contribute to perplexity
  And prompt tokens are masked in loss computation
```

### FR-EVAL-02: Baseline Model Comparison
```gherkin
Scenario: Compare finetuned vs baseline model
  Given a test dataset
  When I run diversity with --model finetuned --baseline-model base
  Then I get perplexity metrics for both models
  And I see improvement percentage
```

### FR-EVAL-03: Generalization Gap
```gherkin
Scenario: Detect overfitting via generalization gap
  Given a train set and test set
  When I run evaluation with --train-set and --test-set
  Then I see train perplexity and test perplexity
  And I see the generalization gap percentage
```

## Implementation Plan

### Phase 1: Core Completion Perplexity
1. Add `compute_completion_perplexity()` to `perplexity.py`
2. Add completion extraction utilities
3. Add `--eval-mode` CLI flag
4. Unit tests for completion extraction

### Phase 2: Model Comparison
1. Create `evaluation.py` module
2. Add `--baseline-model` CLI flag
3. Implement sequential model loading (memory optimization)
4. Integration tests for comparison

### Phase 3: Generalization Evaluation
1. Add `--train-set` and `--test-set` flags
2. Compute generalization gap metrics
3. Add interpretation logic

### Phase 4: Report Integration
1. Add model comparison section to HTML report
2. Add perplexity distribution comparison chart
3. Add generalization gap visualization

## Links

* Depends on: [ADR-004 HTML Diversity Report](./ADR-004-html-diversity-report.md)
* Related: [RFC-001 Dataset Diversity Metrics](../rfcs/RFC-001-dataset-diversity-metrics.md)
