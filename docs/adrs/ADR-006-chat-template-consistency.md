# ADR-006: Chat Template Consistency Across Pipeline

* Status: proposed
* Deciders: Jean
* Date: 2026-01-17

Technical Story: Ensure consistent data formatting between training, inference, and evaluation after implementing model-specific chat templates.

## Context and Problem Statement

We recently implemented chat template support to fix the "endless generation" bug where models finetuned without EOS tokens would generate indefinitely. The solution uses `tokenizer.apply_chat_template()` during training and inference to ensure models see their native format with proper EOS tokens.

However, this change creates format mismatches in other parts of the pipeline that still use the old IRCA format (`### INSTRUCTIONS`, `### ITERATIVE RESOLUTION CYCLE`, etc.).

### Current Data Flow

```
Raw Dataset
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ irca dataset format                                             │
│ - Parses structured columns                                     │
│ - Outputs IRCA-formatted text (### INSTRUCTIONS, ### USER...)   │
│ - Optionally applies augmentations                              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
Formatted Dataset (IRCA format in `text` column)
    │
    ├──────────────────────────────────┬────────────────────────────┐
    ▼                                  ▼                            ▼
┌──────────────────────┐   ┌────────────────────────┐   ┌─────────────────────┐
│ irca finetune run    │   │ irca dataset diversity │   │ Playground/Inference│
│                      │   │                        │   │                     │
│ - Reads `text`       │   │ - Reads `text`         │   │ - Receives query    │
│ - Parses IRCA format │   │ - Computes perplexity  │   │ - Formats as IRCA   │
│ - Converts to msgs   │   │ - Uses IRCA markers    │   │ - Converts to msgs  │
│ - apply_chat_template│   │   for completion split │   │ - apply_chat_template│
│ ✅ Correct format    │   │ ❌ Wrong format        │   │ ✅ Correct format    │
└──────────────────────┘   └────────────────────────┘   └─────────────────────┘
```

### Identified Gaps

#### Gap 1: Diversity Evaluation Format Mismatch

**Problem**: The diversity evaluation computes perplexity on IRCA-formatted text, but:
- Finetuned models were trained on chat-template-formatted text (e.g., Qwen's `<|im_start|>assistant`)
- The model sees a completely different format than what it was trained on

**Impact**: Perplexity values are misleading because they measure how "surprised" the model is by a format it never saw during training.

**Example**:
```python
# What diversity evaluation does (IRCA format):
text = """
### INSTRUCTIONS
You are a helpful assistant.
### USER QUERY
What is 2+2?
### ITERATIVE RESOLUTION CYCLE
Thought: Simple math question...
"""
ppl = compute_perplexity(text, finetuned_model)  # High perplexity!

# What the model was trained on (Qwen format):
text = """
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
What is 2+2?
<|im_end|>
<|im_start|>assistant
Thought: Simple math question...
<|im_end|>
"""
# Much lower perplexity because this matches training!
```

#### Gap 2: Completion Markers Mismatch

**Problem**: The `COMPLETION_MARKERS` in `src/diversity/utils.py` don't cover chat template formats:

```python
COMPLETION_MARKERS = [
    "### ITERATIVE RESOLUTION CYCLE",  # IRCA marker
    "### ASSISTANT",
    "[/INST]",                          # Mistral marker
    "<|assistant|>",                    # ChatML marker
]
```

Missing markers:
- Qwen: `<|im_start|>assistant`
- Llama3: `<|start_header_id|>assistant<|end_header_id|>`
- Model-specific EOS tokens: `<|im_end|>`, `<|eot_id|>`, etc.

**Impact**: Completion-only perplexity evaluation (`--eval-mode completion`) fails to correctly identify where the completion starts.

#### Gap 3: FormatVariationStep Breaks Parsing

**Problem**: The `FormatVariationStep` in `src/formatting/steps.py` randomly replaces markers:

```python
replacements = {
    "### ITERATIVE RESOLUTION CYCLE": "<|ITERATIVE RESOLUTION CYCLE|>",
    "### FINAL ANSWER": "<|FINAL ANSWER|>",
    # ...
}
```

**Impact**: When finetuning reads the `text` column and tries to parse it using `parse_corrected_agent_trace()`, it fails because the markers have been replaced with non-standard versions.

This breaks:
1. `_format_single_example()` in `model_finetuning.py` which looks for `### ITERATIVE RESOLUTION CYCLE`
2. Completion extraction in diversity evaluation

#### Gap 4: Data Storage Strategy

**Architectural Question**: The current approach stores data in IRCA text format and parses it at training time. This has pros and cons:

| Approach | Pros | Cons |
|----------|------|------|
| **Current (IRCA text → parse → messages → chat template)** | Human-readable, familiar format | Parsing overhead, fragile markers, format variations can break parsing |
| **Messages format (store as `[{role, content}, ...]`)** | Clean, no parsing needed, model-agnostic | Requires dataset migration, less human-readable |
| **Pre-formatted per model** | No runtime conversion | Need separate datasets per model family |

## Decision Drivers

* **Accuracy**: Evaluation metrics should reflect actual model performance
* **Consistency**: Same data format seen during training, evaluation, and inference
* **Maintainability**: Avoid fragile string parsing where possible
* **Backward Compatibility**: Existing datasets and workflows should continue to work
* **Model Agnostic**: Support multiple model families (Qwen, Mistral, Llama, etc.)

## Considered Options

### Option 1: Add Chat Template Conversion to Diversity Evaluation

**Description**: Before computing perplexity, convert IRCA text to chat template format using the same conversion pipeline as training.

```python
# In diversity evaluation
def compute_perplexity_with_template(text, model, tokenizer):
    # Parse IRCA format
    parsed = parse_corrected_agent_trace(text)
    # Convert to messages
    messages = irca_to_messages(parsed)
    # Apply model's chat template
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    # Now compute perplexity on correctly formatted text
    return compute_sample_perplexity(formatted, model, tokenizer)
```

| Pros | Cons |
|------|------|
| Minimal dataset changes | Parsing can fail on format-varied data |
| Uses existing infrastructure | Adds complexity to evaluation |
| Accurate perplexity values | Slower evaluation due to parsing |

### Option 2: Store Data in Messages Format

**Description**: Change dataset storage from IRCA text to messages format (list of role/content dicts). Apply chat template at training/evaluation time.

Dataset structure:
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "Thought: Simple math..."}
  ]
}
```

| Pros | Cons |
|------|------|
| Clean, no parsing needed | Requires dataset migration |
| Model-agnostic by design | Less human-readable |
| Robust (no marker fragility) | Breaking change for existing workflows |

### Option 3: Disable FormatVariationStep for Now (Recommended Short-Term)

**Description**: Disable `format_variation` augmentation which breaks parsing. Evaluate Option 1 or 2 for long-term solution.

| Pros | Cons |
|------|------|
| Quick fix | Doesn't solve the fundamental issue |
| No breaking changes | Reduces augmentation variety |
| Buys time for proper solution | |

## Decision Outcome

**Chosen option: Option 3 (Short-Term) + Option 1 (Medium-Term)**

### Phase 1: Immediate Fixes (This Session)

1. **Disable FormatVariationStep by default**: Mark as experimental, document that it breaks chat template conversion
2. **Update COMPLETION_MARKERS**: Add chat template markers for major model families
3. **Document the mismatch**: Users should know evaluation metrics may not reflect training performance

### Phase 2: Medium-Term (Next Sprint)

1. **Add `--apply-chat-template` flag to diversity evaluation**: Optionally convert IRCA → messages → chat template before computing perplexity
2. **Require tokenizer for accurate evaluation**: When `--model` is a finetuned adapter, automatically load its tokenizer and apply chat template

### Phase 3: Long-Term (Future RFC)

1. **Evaluate messages-based storage**: Create RFC to assess migration to messages format
2. **Dataset migration tool**: If messages format is adopted, provide conversion tool

## Detailed Design

### Immediate Changes

#### 1. Update COMPLETION_MARKERS in `src/diversity/utils.py`

```python
COMPLETION_MARKERS = [
    # IRCA format
    "### ITERATIVE RESOLUTION CYCLE",
    "### ASSISTANT",
    "### RESPONSE",
    # Mistral/Llama2
    "[/INST]",
    # ChatML (Qwen, some Llama variants)
    "<|im_start|>assistant",
    "<|assistant|>",
    # Llama3
    "<|start_header_id|>assistant<|end_header_id|>",
]
```

#### 2. Document FormatVariationStep Limitation

Add warning to `src/formatting/steps.py`:

```python
class FormatVariationStep(AugmentationStep):
    """
    Randomize instruction markers and whitespace in formatted text.

    WARNING: This augmentation replaces standard IRCA markers with
    non-standard variants. This BREAKS:
    - Chat template conversion in finetuning
    - Completion extraction in diversity evaluation

    Only use if:
    - Training without chat templates (not recommended)
    - Testing parser robustness (experimental)

    Recommended: Use other augmentations (translate, shuffle_functions)
    that don't break marker parsing.
    """
```

#### 3. Add Chat Template Support to Diversity (Phase 2)

New flag for `irca dataset diversity`:

```bash
irca dataset diversity \
  -d datasets/formatted \
  --model ./finetuned-adapter \
  --apply-chat-template  # NEW: Convert IRCA → chat template before perplexity
```

Implementation in `src/diversity/perplexity.py`:

```python
def compute_perplexity_profile(
    texts: list[str],
    model_name: str,
    # ... existing params ...
    apply_chat_template: bool = False,  # NEW
) -> dict[str, Any]:
    """
    Args:
        apply_chat_template: If True, convert IRCA-formatted text to model's
                            native chat template before computing perplexity.
                            Recommended for evaluating finetuned models.
    """
    model, tokenizer = load_model_and_tokenizer(model_name)

    for text in texts:
        if apply_chat_template:
            text = convert_irca_to_chat_template(text, tokenizer)
        ppl = compute_sample_perplexity(text, model, tokenizer)
        # ...
```

## Consequences

### Positive

* **Accurate evaluation**: Perplexity metrics reflect actual model performance
* **Consistent pipeline**: Same format used in training, evaluation, and inference
* **Clear documentation**: Users understand the format mismatch issue
* **Incremental approach**: Can improve progressively without breaking changes

### Negative

* **Reduced augmentation variety**: FormatVariationStep discouraged
* **Added complexity**: Chat template conversion in evaluation pipeline
* **Tokenizer dependency**: Evaluation now requires knowing which model will be used

### Neutral

* **Existing workflows**: Continue to work but with documented limitations
* **Messages format**: Deferred decision, may be addressed in future RFC

## Related Issues

* Endless generation bug (fixed by chat templates)
* ADR-003: Dataset Format Pipeline
* ADR-005: Completion Perplexity Evaluation

## Requirements

| Req ID | Name | Summary |
|--------|------|---------|
| FR-EVAL-01 | Chat Template Perplexity | Compute perplexity on chat-template-formatted text |
| FR-EVAL-02 | Model-Specific Completion Markers | Detect completion start for multiple model families |
| FR-DATA-29 | Safe Augmentations | Augmentations must not break parsing/conversion |

## Acceptance Tests

### FR-EVAL-01: Chat Template Perplexity
```gherkin
Scenario: Perplexity with chat template conversion
  Given a dataset formatted in IRCA format
  And a model finetuned with Qwen chat templates
  When I run diversity evaluation with --apply-chat-template
  Then the text is converted to Qwen format before perplexity computation
  And perplexity values reflect actual model performance
```

### FR-DATA-29: Safe Augmentations
```gherkin
Scenario: Augmented data can be parsed
  Given a dataset formatted with augmentations
  When I run finetuning on the augmented dataset
  Then all samples are successfully parsed to messages
  And chat templates are correctly applied
```

## Implementation Checklist

- [ ] Update COMPLETION_MARKERS with chat template markers
- [ ] Add warning to FormatVariationStep documentation
- [ ] Add `--apply-chat-template` flag to diversity command (Phase 2)
- [ ] Create RFC for messages-based storage (Phase 3)
- [ ] Update EXPERIENCES.md with learnings
