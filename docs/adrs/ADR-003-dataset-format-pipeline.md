# ADR-003: Dataset Format Pipeline for Evaluation and Training

* Status: proposed
* Deciders: Jean
* Date: 2026-01-16

Technical Story: Enable clean separation of raw dataset → formatting → augmentation for both diversity evaluation and finetuning

## Context and Problem Statement

The current data pipeline conflates **formatting** (converting structured data to raw text) with **augmentation** (applying transformations). This creates several problems:

### Current Architecture Issues

1. **No clean baseline for comparison**: Raw datasets have structured columns (`corrected_agent_trace`, `user_query`, etc.) but no `text` column. The diversity evaluation tool requires a `text` column.

2. **Formatting + Augmentation are coupled**: The `augment` command simultaneously:
   - Parses `corrected_agent_trace[0]['value']` into parts
   - Optionally translates user_query and final_answer
   - Rebuilds into `text` column using `build_full_prompt()`

3. **`augment-pipeline` expects pre-formatted data**: The pipeline steps (`shuffle_functions`, `format_variation`, etc.) operate on the `text` column, requiring data to already be formatted.

4. **Cannot compare original vs augmented for diversity**: To measure augmentation effectiveness, we need:
   - Formatted baseline (no augmentation)
   - Formatted + augmented version
   - But there's no way to produce a "formatted baseline" without augmentation

### Data Flow Diagram (Current State)

```
Raw Dataset                      Augmented Dataset
┌────────────────────────┐       ┌─────────────────────────────────────┐
│ corrected_agent_trace  │       │ corrected_agent_trace (unchanged)   │
│ user_query             │ ────► │ user_query (unchanged)              │
│ available_functions    │       │ text (formatted + maybe translated) │
│ ...                    │       │ ...                                 │
└────────────────────────┘       └─────────────────────────────────────┘
        │                                        │
        │  `irca dataset augment`                │
        │  (formats + translates in one step)    │
        ▼                                        ▼
                                  ┌─────────────────────────────────────┐
                                  │ `irca dataset augment-pipeline`     │
                                  │ (requires `text` column)            │
                                  │ (applies presentation augmentations)│
                                  └─────────────────────────────────────┘
```

### The Missing Piece

We cannot produce a **formatted-only** dataset (no augmentation) to serve as a baseline for:
1. Diversity comparison (original vs augmented)
2. Perplexity evaluation (what the model actually sees)
3. Training without augmentation (ablation studies)

## Decision Drivers

* **Separation of concerns**: Formatting and augmentation should be independent operations
* **Reproducibility**: Same formatting logic for training and evaluation
* **Flexibility**: Ability to compare any combination of augmentations
* **Simplicity**: Single source of truth for "how raw data becomes training text"
* **Backward compatibility**: Existing workflows should continue to work

## Considered Options

### Option 1: Add `--no-augment` Flag to Existing `augment` Command

**Description**: Modify `irca dataset augment` to support a "format only" mode.

```bash
# Format only (new)
irca dataset augment -i raw-dataset -o formatted --no-augment

# Format + translate (existing)
irca dataset augment -i raw-dataset -o augmented -l fr -r 0.5
```

| Pros | Cons |
|------|------|
| Minimal changes | Confusing semantics ("augment" without augmenting) |
| Reuses existing code | `--no-augment` conflicts with required `-l` languages |
| | Doesn't solve pipeline integration |

### Option 2: New `format` Command (Recommended)

**Description**: Create a dedicated `irca dataset format` command that:
1. Converts raw structured data → `text` column
2. Optionally applies augmentation via config file
3. Produces datasets ready for training or evaluation

```bash
# Format baseline (no augmentation)
irca dataset format -i raw-dataset -o formatted-baseline

# Format with augmentation config
irca dataset format -i raw-dataset -o formatted-augmented -c augment.json

# Compare diversity
irca dataset diversity -o formatted-baseline -a formatted-augmented --quick
```

| Pros | Cons |
|------|------|
| Clear semantics | New command to learn |
| Clean separation of concerns | Some code duplication with `augment` |
| Single entry point for raw → text | |
| Composable with diversity evaluation | |

### Option 3: Integrate Formatting into `augment-pipeline`

**Description**: Make `augment-pipeline` handle raw datasets by adding a formatting stage.

```json
{
  "format": {
    "source_column": "corrected_agent_trace",
    "apply_formatting_augmentations": true
  },
  "multiply": 2,
  "pipeline": [...]
}
```

| Pros | Cons |
|------|------|
| Single command for everything | More complex config |
| Flexible | Harder to produce "format only" baseline |
| | Config becomes mandatory even for simple formatting |

## Decision Outcome

**Chosen option: Option 2 - New `format` Command**

This provides the cleanest separation of concerns and enables the desired workflows for diversity evaluation while maintaining simplicity.

## Detailed Design

### New Command: `irca dataset format`

```bash
irca dataset format [OPTIONS]

Options:
  -i, --input TEXT           Input dataset (local path or HuggingFace ID) [required]
  -o, --output PATH          Output path for formatted dataset [required]
  -c, --config PATH          Augmentation config file (optional)
  --source-column TEXT       Column containing raw prompt [default: corrected_agent_trace]
  --no-augment              Disable all augmentations (even in config)
  --seed INTEGER            Random seed for augmentations [default: 42]
  --help                    Show this message and exit.
```

### Data Flow (New Architecture)

```
                              ┌─────────────────────────────────────┐
                              │         irca dataset format         │
Raw Dataset                   │                                     │
┌──────────────────────┐      │  1. Parse source column             │
│ corrected_agent_trace│──────│  2. Extract parts (system, query,   │
│ user_query           │      │     functions, completion)          │
│ available_functions  │      │  3. If config: apply augmentations  │
│ ...                  │      │  4. Rebuild as `text` column        │
└──────────────────────┘      └───────────────┬─────────────────────┘
                                              │
                              ┌───────────────▼─────────────────────┐
                              │       Formatted Dataset              │
                              │  (has `text` column, ready for:)     │
                              │  - Finetuning                        │
                              │  - Diversity evaluation              │
                              │  - Perplexity computation            │
                              └─────────────────────────────────────┘
```

### Augmentation Stages

The `format` command processes augmentations in three stages:

```
┌──────────────────────────────────────────────────────────────────┐
│                    AUGMENTATION STAGES                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: STRUCTURAL (before formatting)                         │
│  ─────────────────────────────────────────                       │
│  Operates on parsed components (user_query, completion, etc.)    │
│  • translate_query: Translate user_query                         │
│  • translate_answer: Translate final answer                      │
│                                                                  │
│  Stage 2: FORMATTING                                             │
│  ────────────────────                                            │
│  Converts parsed components → text using build_full_prompt()     │
│  • function_shuffle: Shuffle function order in JSON              │
│  • function_indent: Randomize JSON indentation                   │
│                                                                  │
│  Stage 3: PRESENTATION (after formatting)                        │
│  ─────────────────────────────────────────                       │
│  Operates on final text                                          │
│  • format_variation: Randomize section markers                   │
│  • newline_variation: Randomize \n vs \r\n                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Configuration Format

The format command uses the same JSON config as `augment-pipeline`, but with explicit stage markers:

```json
{
  "seed": 42,
  "structural": [
    {
      "type": "translate",
      "probability": 0.5,
      "params": {
        "languages": ["fr", "es", "de"],
        "targets": ["query", "answer"]
      }
    }
  ],
  "formatting": [
    {
      "type": "shuffle_functions",
      "probability": 0.8
    },
    {
      "type": "indent_functions",
      "probability": 0.5,
      "params": {"choices": [null, 2, 4]}
    }
  ],
  "presentation": [
    {
      "type": "format_variation",
      "probability": 0.3
    },
    {
      "type": "newline_variation",
      "probability": 0.2
    }
  ]
}
```

### Workflow: Diversity Evaluation

```bash
# 1. Format baseline (no augmentation)
irca dataset format \
  -i datasets/irca-raw \
  -o datasets/formatted-baseline \
  --no-augment

# 2. Format with augmentation
irca dataset format \
  -i datasets/irca-raw \
  -o datasets/formatted-augmented \
  -c configs/augment_full.json

# 3. Compare diversity
irca dataset diversity \
  -o datasets/formatted-baseline \
  -a datasets/formatted-augmented \
  --quick
```

### Workflow: Dataset Multiplication + Formatting

For dataset size multiplication (ADR-002), the workflow becomes:

```bash
# Option A: Format first, then multiply
irca dataset format -i raw -o formatted --no-augment
irca dataset augment-pipeline -c multiply.json -i formatted -o multiplied

# Option B: Format + multiply in one step (using multiply in config)
irca dataset format -i raw -o multiplied -c multiply_and_augment.json
```

Config for Option B:
```json
{
  "seed": 42,
  "multiply": 3,
  "structural": [...],
  "formatting": [...],
  "presentation": [...]
}
```

### Output Dataset Structure

The output dataset will have:

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | Formatted prompt (ready for training) |
| `original_index` | int | Index in source dataset |
| `variant_id` | int | 0 = baseline, 1+ = augmented variant |
| `augmentations` | list[str] | Applied augmentation names |
| (original columns) | various | Preserved from input |

### Integration with Existing Commands

| Command | Change |
|---------|--------|
| `irca dataset augment` | **Deprecated** - Use `format` with translation config |
| `irca dataset augment-pipeline` | **Deprecated** - Use `format` with full config |
| `irca dataset diversity` | No change - works with `text` column |
| `irca finetune run` | No change - expects `text` column |

### Migration Path

```bash
# Old way (deprecated)
irca dataset augment -i raw -o augmented -l fr -r 0.5

# New way (equivalent)
# Create config: translate_fr.json
# {
#   "seed": 42,
#   "structural": [
#     {"type": "translate", "probability": 0.5, "params": {"languages": ["fr"]}}
#   ]
# }
irca dataset format -i raw -o augmented -c translate_fr.json
```

## Consequences

### Positive

* **Clean separation**: Formatting is now an explicit, independent step
* **Reproducible baselines**: Can always produce non-augmented formatted data
* **Unified workflow**: Same command for training prep and evaluation prep
* **Diversity evaluation enabled**: Can compare any augmentation configuration
* **Ablation studies**: Easy to test impact of specific augmentations

### Negative

* **Two commands deprecated**: Users must migrate to new workflow
* **Config file required for augmentation**: No more inline `-l fr` shorthand
* **Learning curve**: New command and concepts to understand

### Neutral

* **Existing datasets with `text` column**: Still work with finetuning/diversity
* **`augment-pipeline` still works**: For users who already have formatted data

## Implementation Plan

### Phase 1: Core Format Command
1. Create `src/formatting/` module
2. Implement `DatasetFormatter` class with stage-based processing
3. Add `irca dataset format` CLI command
4. Support `--no-augment` for baseline generation

### Phase 2: Augmentation Integration
1. Move structural augmentations (translate) to new stage system
2. Move formatting augmentations (shuffle, indent) to new stage system
3. Move presentation augmentations (markers, newlines) to new stage system
4. Create config schema for staged augmentations

### Phase 3: Multiply Support
1. Add `multiply` option to format config
2. Implement variant generation during formatting
3. Add deduplication support (reuse from pipeline)

### Phase 4: Migration & Cleanup
1. Add deprecation warnings to old commands
2. Update documentation and examples
3. Create migration guide
4. Update diversity evaluation examples

## Requirements

This ADR introduces the following requirements:

| Req ID | Name | Summary |
|--------|------|---------|
| FR-DATA-24 | Dataset Format Command | Format raw structured data to training-ready text |
| FR-DATA-25 | Staged Augmentation | Process augmentations in structural/formatting/presentation stages |
| FR-DATA-26 | Format Baseline | Generate formatted baseline without augmentations |
| FR-DATA-27 | Format Config | JSON configuration for format augmentations |
| FR-DATA-28 | Diversity Evaluation Workflow | Enable original vs augmented comparison |

## Acceptance Tests

### FR-DATA-24: Dataset Format Command
```gherkin
Scenario: Format raw dataset to text
  Given a dataset with 'corrected_agent_trace' column
  When I run "irca dataset format -i raw -o formatted --no-augment"
  Then the output dataset has a 'text' column
  And the 'text' column contains properly formatted prompts
  And all original columns are preserved
```

### FR-DATA-26: Format Baseline
```gherkin
Scenario: Baseline is reproducible
  Given a raw dataset
  When I run format with --no-augment twice
  Then both outputs have identical 'text' content
  And no random variations are applied

Scenario: Baseline works with diversity
  Given a formatted baseline dataset
  When I run "irca dataset diversity -d baseline --quick"
  Then diversity metrics are computed successfully
```

### FR-DATA-28: Diversity Evaluation Workflow
```gherkin
Scenario: Compare original vs augmented diversity
  Given a raw dataset
  When I format without augmentation to 'baseline'
  And I format with augmentation config to 'augmented'
  And I run diversity comparison between them
  Then I see lexical diversity metrics for both
  And I see the delta between them
```

## Links

* Depends on: [ADR-002 Dataset Size Multiplication](./ADR-002-dataset-size-multiplication.md)
* Related: [RFC-001 Dataset Diversity Metrics](../rfcs/RFC-001-dataset-diversity-metrics.md)
* Traceability: [TRACEABILITY_MATRIX.md](../TRACEABILITY_MATRIX.md)
