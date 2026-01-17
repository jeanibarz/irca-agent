# Dataset Size Multiplication with Randomized Augmentation

* Status: proposed
* Deciders: Jean
* Date: 2026-01-16

Technical Story: Enable controlled dataset expansion through randomized augmentation features

## Context and Problem Statement

The current `irca dataset augment` command applies translation to a **ratio** of samples but does not change the dataset size. For example, with `--ratio 0.5`, 50% of 1000 samples get translated, but the output still has 1000 samples.

For effective finetuning, we often need to **multiply the dataset size** (e.g., x3 means 1000 samples become 3000) while applying diverse augmentations. This requires a new approach that:

1. Controls the **output dataset size** relative to input
2. Applies **randomized augmentation features** to generate variants
3. Maintains **semantic correctness** (preserves tool calls, Output references)
4. Ensures **reproducibility** via seeded randomness

### Existing Augmentation Features

The codebase already has several augmentation techniques that can be leveraged:

| Feature | Location | Description |
|---------|----------|-------------|
| **Function Shuffling** | `src/core/utils.py:120-134` | Randomizes order of functions in JSON to reduce positional bias |
| **Function Removal** | `src/core/generation/trace_generator.py:217-296` | Creates traces with missing functions to teach capability gap handling |
| **Newline Randomization** | `src/core/prompt_builder.py:133-147` | Randomly uses `\n` or `\r\n` for formatting robustness |
| **Instruction Formatting** | `src/core/prompt_builder.py:150-180` | Randomizes markers (`### INSTRUCTIONS` → `<\|FUNCTIONS AVAILABLE\|>`) and whitespace |
| **Translation** | `src/dataset_generation/translator.py` | Translates user queries and final answers to target languages |

These features are currently applied:
- **At generation time**: Function removal creates alternative traces
- **At training time**: Shuffling and formatting via `InstructionFormatter(random_augmentation=True)`
- **At dataset prep time**: Translation via `irca dataset augment`

### Current Behavior

```bash
# Input: 1000 samples
# Output: 1000 samples (50% translated to French)
irca dataset augment -i dataset -o augmented -l fr -r 0.5
```

### Desired Behavior

```bash
# Input: 1000 samples
# Output: 3000 samples (original + 2 augmented variants per sample)
irca dataset augment -i dataset -o augmented -l fr es de --multiply 3
```

## Decision Drivers

* **Flexibility**: Support various augmentation strategies and multipliers
* **Quality**: Generate meaningful variants, not just duplicates
* **Efficiency**: Minimize redundant computation (model loading, translation)
* **Simplicity**: Easy to understand and configure
* **Reproducibility**: Deterministic output given the same seed
* **Extensibility**: Support future augmentation types beyond translation

## Considered Options

### Option 1: Multi-Language Round-Robin

**Description**: Generate N variants per sample by cycling through available languages.

```
Original → [Original, French, Spanish, German, French, ...]
```

**Algorithm**:
```python
for sample in dataset:
    yield sample  # Always keep original
    for i in range(multiplier - 1):
        lang = languages[i % len(languages)]
        yield translate(sample, lang)
```

**Example** (x3, languages=[fr, es]):
- Sample 1 → [Original, French, Spanish]
- Sample 2 → [Original, French, Spanish]

| Pros | Cons |
|------|------|
| Simple to implement | Predictable/deterministic distribution |
| Balanced language distribution | Limited to number of languages |
| Easy to understand | No randomization within samples |
| Efficient batching by language | All samples get same augmentations |

### Option 2: Stochastic Multi-Variant Generation

**Description**: For each variant, randomly select augmentation features from a pool.

```
Original → [Original, Random(features), Random(features), ...]
```

**Algorithm**:
```python
for sample in dataset:
    yield sample  # Always keep original
    for i in range(multiplier - 1):
        features = random.choice(augmentation_pool)
        yield apply_augmentation(sample, features)
```

**Example** (x3, pool=[fr, es, de, paraphrase]):
- Sample 1 → [Original, Spanish, French]
- Sample 2 → [Original, German, Spanish]
- Sample 3 → [Original, French, French]  # Duplicates possible!

| Pros | Cons |
|------|------|
| High variance in output | May generate duplicate variants |
| Natural distribution | Less predictable language balance |
| Supports mixed augmentation types | Harder to debug specific variants |
| More realistic training diversity | May need deduplication |

### Option 3: Cartesian Product Expansion

**Description**: Generate all combinations of augmentation features.

```
Original → [Original, Fr, Es, De, Fr+Paraphrase, Es+Paraphrase, ...]
```

**Algorithm**:
```python
combinations = list(itertools.product(languages, styles, ...))
for sample in dataset:
    yield sample
    for combo in combinations[:multiplier-1]:
        yield apply_augmentations(sample, combo)
```

**Example** (languages=[fr, es], styles=[formal, casual]):
- Generates: Original, Fr-Formal, Fr-Casual, Es-Formal, Es-Casual (x5)

| Pros | Cons |
|------|------|
| Systematic coverage | Exponential growth (combinatorial explosion) |
| No duplicates | Fixed multiplier based on feature count |
| Predictable output | Cannot easily target specific multiplier |
| Good for small feature sets | Overkill for simple augmentation |

### Option 4: Weighted Feature Sampling (Recommended)

**Description**: Sample augmentation features with configurable weights, guaranteeing unique variants per sample.

```
Original → [Original, WeightedSample(features), WeightedSample(features), ...]
```

**Algorithm**:
```python
for sample in dataset:
    yield sample  # Always keep original
    used_features = set()
    for i in range(multiplier - 1):
        # Sample without replacement within each sample
        available = [f for f in features if f not in used_features]
        if not available:
            available = features  # Reset if exhausted
        feature = weighted_choice(available, weights)
        used_features.add(feature)
        yield apply_augmentation(sample, feature)
```

**Example** (x3, features={fr: 0.4, es: 0.4, de: 0.2}):
- Sample 1 → [Original, French, Spanish]  # No duplicates within sample
- Sample 2 → [Original, Spanish, German]
- Sample 3 → [Original, French, German]

| Pros | Cons |
|------|------|
| Controlled distribution via weights | More complex configuration |
| No duplicates within a sample | Weights require tuning |
| Flexible multiplier | Slightly more complex implementation |
| Supports any augmentation type | |
| Reproducible with seed | |

### Option 5: Stratified Expansion

**Description**: Divide dataset into strata, apply different augmentations to each stratum.

```
Stratum A (30%) → French only
Stratum B (30%) → Spanish only
Stratum C (40%) → Original only
```

**Algorithm**:
```python
strata = split_dataset(dataset, ratios=[0.3, 0.3, 0.4])
for stratum, augmentation in zip(strata, [fr, es, None]):
    for sample in stratum:
        yield sample
        if augmentation and need_more_samples():
            yield apply_augmentation(sample, augmentation)
```

| Pros | Cons |
|------|------|
| Clear separation of augmentation types | Complex to configure |
| Good for A/B testing augmentations | Uneven sample treatment |
| Easy to analyze per-stratum | Doesn't naturally support multiplier |
| | Hard to combine augmentations |

### Option 6: Iterative Pipeline

**Description**: Chain multiple augmentation stages, each can expand the dataset.

```
Stage 1: x2 via languages → Stage 2: x1.5 via paraphrasing → Total: x3
```

**Algorithm**:
```python
dataset = original_dataset
for stage in augmentation_stages:
    dataset = stage.expand(dataset, stage.multiplier)
# Final size = original * product(stage.multipliers)
```

| Pros | Cons |
|------|------|
| Composable stages | Order of stages matters |
| Each stage is simple | Multiplier math is multiplicative |
| Supports complex pipelines | Slow (multiple passes) |
| | Intermediate datasets consume memory |

## Decision Outcome

**Chosen option: Pipeline-based Transformation with Independent Probabilities**

A hybrid approach combining the flexibility of Option 4 (weighted sampling) with the composability of Option 6 (pipeline). Each augmentation feature:
- Has its own **independent probability** of being applied
- Can be **combined** with other features (applied sequentially)
- Is defined in a **JSON configuration file** for reproducibility and versioning

### Core Concepts

1. **Multiplier**: Controls output dataset size (e.g., x3 = 3000 samples from 1000)
2. **Pipeline**: Ordered list of transformation steps, each with independent probability
3. **Combination**: Multiple features can be applied to the same variant (sequentially)
4. **Original preservation**: Original sample is always kept (variant_id=0)

### JSON Configuration File

```json
{
  "input": "datasets/irca-v1",
  "output": "datasets/irca-v1-augmented",
  "multiply": 3,
  "seed": 42,
  "pipeline": [
    {
      "type": "translate",
      "probability": 0.5,
      "params": {
        "languages": ["fr", "es", "de"],
        "weights": [0.4, 0.4, 0.2]
      }
    },
    {
      "type": "shuffle_functions",
      "probability": 0.8
    },
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

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `input` | string | Yes* | - | Input dataset path (can be overridden via CLI) |
| `output` | string | Yes* | - | Output dataset path (can be overridden via CLI) |
| `multiply` | int | Yes | - | Size multiplier (e.g., 3 = 3x samples) |
| `seed` | int | Yes | - | Random seed for reproducibility |
| `deduplicate` | bool/string | No | `true` | Remove duplicate variants (`true`, `false`, or `"fuzzy"`) |
| `pipeline` | array | Yes | - | Ordered list of augmentation steps |

*Required in config OR via CLI `--input`/`--output` flags.

### How It Works

For each variant to generate:
1. Start with the original sample
2. Walk through the pipeline **sequentially**
3. For each step, roll against its probability
4. If triggered, apply the transformation
5. Record which transformations were applied

**Example** with config above (x3 multiplier):
```
Sample 1:
  - Variant 0: Original (always kept)
  - Variant 1: [translate:fr, shuffle_functions] (rolled: translate=yes→fr, shuffle=yes, format=no, newline=no)
  - Variant 2: [shuffle_functions, format_variation] (rolled: translate=no, shuffle=yes, format=yes, newline=no)

Sample 2:
  - Variant 0: Original
  - Variant 1: [translate:es, shuffle_functions, newline_variation]
  - Variant 2: [shuffle_functions]
```

### CLI Interface

**Simplified**: Config file is required. All augmentation parameters live in the config.

```bash
# Standard usage
irca dataset augment --config configs/augment_multilingual.json

# Override input/output paths (useful for reusing configs)
irca dataset augment --config configs/augment_multilingual.json \
    --input datasets/irca-v2 \
    --output datasets/irca-v2-augmented

# With W&B tracking
irca dataset augment --config configs/augment_full.json --track-wandb
```

### Proposed Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--config` / `-c` | Path | **Yes** | JSON configuration file (contains all augmentation settings) |
| `--input` / `-i` | Path | No | Override input path from config |
| `--output` / `-o` | Path | No | Override output path from config |
| `--track-wandb` | Flag | No | Log to W&B as artifact |
| `--run-name` | str | No | Custom W&B run name |

**Removed arguments** (now in config):
- `--multiply` → `config.multiply`
- `--languages` → `config.pipeline[].params.languages`
- `--ratio` → removed (use `probability` in pipeline steps)
- `--seed` → `config.seed`

**Benefits of config-only approach**:
1. **Single source of truth**: All settings in one versioned file
2. **Reproducibility**: Config file is copied to output metadata
3. **Simpler CLI**: Only 2 required concepts (config + paths)
4. **Reusable configs**: Same config works with different datasets via `--input`/`--output` overrides
5. **Self-documenting**: Config file serves as documentation of what was done

### Backward Compatibility

The existing ratio-based mode (`-l fr -r 0.5`) will be **deprecated** in favor of config files.

**Migration path**:
```bash
# Old (deprecated)
irca dataset augment -i ds -o ds-aug -l fr -l es -r 0.5

# New (equivalent config)
# configs/legacy_ratio.json:
# {
#   "multiply": 1,
#   "seed": 42,
#   "pipeline": [
#     {"type": "translate", "probability": 0.5, "params": {"languages": ["fr", "es"]}}
#   ]
# }
irca dataset augment -c configs/legacy_ratio.json -i ds -o ds-aug
```

**Note**: `multiply: 1` with translation probability produces the same effect as the old ratio mode (same dataset size, portion translated).

### Available Pipeline Steps

| Type | Parameters | Description | Source |
|------|------------|-------------|--------|
| `translate` | `languages`: list, `weights`: list (optional) | Translate user query and final answer | `translator.py` |
| `shuffle_functions` | None | Randomize function order in JSON | `utils.py:120` |
| `format_variation` | None | Randomize instruction markers and whitespace | `prompt_builder.py:150` |
| `newline_variation` | None | Randomize `\n` vs `\r\n` | `prompt_builder.py:133` |

**Future steps** (not yet implemented):
| Type | Parameters | Description |
|------|------------|-------------|
| `paraphrase` | `style`: formal/casual/technical | Rephrase using LLM |
| `typo_injection` | `rate`: 0.0-1.0 | Add realistic typos |
| `query_rephrase` | None | Rephrase user query |

### Output Dataset Structure

Each output sample includes metadata tracking all applied transformations:

```python
{
    "text": "...",  # Augmented prompt
    "original_index": 42,  # Index in source dataset
    "variant_id": 1,  # 0 = original, 1+ = augmented variants
    "augmentations": ["translate:fr", "shuffle_functions"],  # List of applied transforms
}
```

### Reproducibility & Experiment Tracking

For full reproducibility, the augmentation process must capture and store comprehensive metadata. The project already uses **W&B (Weights & Biases)** for finetuning tracking, so we'll extend this to augmentation.

#### Metadata to Capture

| Category | Fields | Purpose |
|----------|--------|---------|
| **Environment** | `commit_hash`, `branch`, `dirty` (uncommitted changes) | Code version |
| **Configuration** | Full JSON config content, CLI arguments | Exact parameters |
| **Input** | `input_path`, `input_hash` (SHA256), `input_size` | Source dataset identity |
| **Output** | `output_path`, `output_size`, `timestamp` | Result identity |
| **Runtime** | `python_version`, `package_versions`, `duration_seconds` | Environment details |
| **Statistics** | `augmentation_distribution`, `languages_used`, `empty_variant_rerolls` | Quality metrics |

#### Storage Strategy: Layered Approach

**Layer 1: Dataset Sidecar File** (always generated)

Store a `_augmentation_metadata.json` alongside the output dataset:

```json
{
  "schema_version": "1.0",
  "timestamp": "2026-01-16T14:30:00Z",
  "git": {
    "commit_hash": "abc123def456",
    "branch": "main",
    "dirty": false
  },
  "config": {
    "multiply": 3,
    "seed": 42,
    "pipeline": [...]
  },
  "input": {
    "path": "datasets/irca-v1",
    "hash": "sha256:abc123...",
    "num_samples": 1000
  },
  "output": {
    "path": "datasets/irca-v1-augmented",
    "num_samples": 3000
  },
  "statistics": {
    "augmentation_counts": {
      "translate:fr": 450,
      "translate:es": 430,
      "shuffle_functions": 1800,
      "format_variation": 600
    },
    "empty_variant_rerolls": 12,
    "duration_seconds": 342.5
  },
  "environment": {
    "python_version": "3.10.12",
    "irca_version": "0.2.0",
    "key_packages": {
      "transformers": "4.40.0",
      "datasets": "2.18.0",
      "torch": "2.2.0"
    }
  }
}
```

**Layer 2: W&B Artifact** (optional, when W&B configured)

Log the augmented dataset as a W&B Artifact with metadata:

```python
import wandb

# Create artifact
artifact = wandb.Artifact(
    name="irca-dataset-augmented",
    type="dataset",
    metadata={
        "config": config,
        "input_hash": input_hash,
        "statistics": statistics,
    }
)

# Add dataset files
artifact.add_dir(output_path)

# Log to W&B
wandb.log_artifact(artifact)
```

Benefits:
- **Searchable**: Query datasets by metadata in W&B UI
- **Linked**: Connect augmented datasets to finetuning runs
- **Versioned**: W&B tracks artifact versions automatically
- **Shareable**: Team members can access via W&B

**Layer 3: HuggingFace Dataset Info** (embedded in dataset)

Use the built-in `DatasetInfo` mechanism:

```python
from datasets import Dataset, DatasetInfo

dataset_info = DatasetInfo(
    description="Augmented IRCA dataset",
    version="1.0.0",
    license="...",
    # Custom fields in features metadata
)

dataset = Dataset.from_dict(data, info=dataset_info)
```

#### CLI Integration

```bash
# Basic: Only sidecar file (default)
irca dataset augment -i dataset -o augmented --config config.json

# With W&B tracking
irca dataset augment -i dataset -o augmented --config config.json --track-wandb

# With custom run name
irca dataset augment -i dataset -o augmented --config config.json --track-wandb --run-name "aug-v3-multilingual"
```

#### Linking Augmentation to Finetuning

When finetuning uses an augmented dataset, reference the augmentation metadata:

```bash
# Finetune automatically detects and logs augmentation metadata
irca finetune run --dataset datasets/irca-v1-augmented --model Qwen/Qwen2.5-1.5B
```

The finetuning run logs:
- Reference to augmentation W&B artifact (if available)
- Copy of `_augmentation_metadata.json` content
- Link between augmentation config and training results

This enables **full lineage tracking**: raw data → augmentation config → augmented data → training config → model.

### Algorithm Pseudocode

```python
def augment_with_pipeline(dataset, config):
    rng = random.Random(config["seed"])
    multiplier = config["multiply"]
    pipeline = config["pipeline"]
    deduplicate = config.get("deduplicate", True)

    # Phase 1: Generate all variants
    all_samples = []
    for idx, sample in enumerate(dataset):
        # Always emit original
        all_samples.append({
            **sample,
            "original_index": idx,
            "variant_id": 0,
            "augmentations": [],
        })

        # Generate (multiplier - 1) variants
        for variant_id in range(1, multiplier):
            augmented = sample.copy()
            applied = []

            # Walk through pipeline sequentially
            for step in pipeline:
                if rng.random() < step["probability"]:
                    augmented = apply_step(augmented, step, rng)
                    applied.append(format_step_name(step))

            all_samples.append({
                **augmented,
                "original_index": idx,
                "variant_id": variant_id,
                "augmentations": applied,
            })

    # Phase 2: Deduplicate if enabled
    if deduplicate:
        all_samples, stats = deduplicate_variants(all_samples)

    return all_samples, stats


def deduplicate_variants(samples):
    """Remove duplicate variants within each original sample."""
    seen_per_original = defaultdict(set)
    deduplicated = []
    duplicates_removed = 0

    for sample in samples:
        text_hash = hash(sample["text"])
        original_idx = sample["original_index"]

        if text_hash in seen_per_original[original_idx]:
            duplicates_removed += 1
            continue

        seen_per_original[original_idx].add(text_hash)
        deduplicated.append(sample)

    return deduplicated, {"duplicates_removed": duplicates_removed}


def apply_step(sample, step, rng):
    """Apply a single pipeline step to a sample."""
    match step["type"]:
        case "translate":
            lang = weighted_choice(step["params"]["languages"],
                                   step["params"].get("weights"), rng)
            return translate_sample(sample, lang)
        case "shuffle_functions":
            return shuffle_functions_in_sample(sample, rng)
        case "format_variation":
            return apply_format_variation(sample, rng)
        case "newline_variation":
            return apply_newline_variation(sample, rng)
        case _:
            raise ValueError(f"Unknown step type: {step['type']}")
```

### Pipeline Order Considerations

The order of steps in the pipeline matters for some combinations:

| Order | Effect |
|-------|--------|
| translate → shuffle | Translate first, then shuffle (functions stay in original language names) |
| shuffle → translate | Shuffle first, then translate (same result for function names) |
| format_variation → translate | Format markers changed, then text translated |
| translate → format_variation | Text translated, then markers randomized |

**Recommendation**: Place `translate` first since it's the most expensive operation and other transformations are lightweight text manipulations.

### Positive Consequences

* **Full flexibility**: Any combination of features possible
* **Independent control**: Each feature has its own probability
* **Reproducible**: JSON config can be versioned and shared
* **Composable**: Easy to add new pipeline steps
* **Traceable**: Metadata records exactly what was applied
* **Backward compatible**: Ratio mode still works

### Negative Consequences

* **Variant uniqueness not guaranteed**: Two variants might get the same combination by chance
* **Config complexity**: JSON file adds configuration overhead
* **Pipeline order matters**: Users must understand transformation interactions
* **Empty variants possible**: If all probabilities are low, a variant might have no transformations

## Handling Duplicates

### When Duplicates Can Occur

| Scenario | Result |
|----------|--------|
| Two variants with no augmentations | Identical to original |
| Two variants with only `translate:fr` | Same French text (translation is deterministic) |
| Two variants with `shuffle_functions` | Different (random shuffle each time) |
| Two variants with `format_variation` | Different (random choices each time) |

Duplicates are **possible but rare**, mainly from empty variants or same translation-only combinations.

### Solution: Post-Generation Deduplication

Instead of complex re-roll logic, add a simple **deduplication step at the end**:

```json
{
  "multiply": 3,
  "seed": 42,
  "deduplicate": true,
  "pipeline": [...]
}
```

**Algorithm:**
```python
def deduplicate_variants(samples):
    """Remove duplicate variants within each original sample."""
    seen_per_original = defaultdict(set)

    for sample in samples:
        text_hash = hash(sample["text"])
        original_idx = sample["original_index"]

        if text_hash in seen_per_original[original_idx]:
            continue  # Skip duplicate

        seen_per_original[original_idx].add(text_hash)
        yield sample
```

**Benefits of deduplication approach:**
1. **Simpler pipeline**: No re-roll logic, no retry limits
2. **Handles all cases**: Empty variants, same translations, any edge case
3. **Transparent**: Metadata reports how many duplicates removed
4. **Optional**: Can disable if duplicates are acceptable for training

**Config options:**
| Value | Behavior |
|-------|----------|
| `true` or `"exact"` | Remove exact text duplicates |
| `false` (default) | Keep all variants including duplicates |
| `"fuzzy"` | Future: Remove near-duplicates (similarity threshold) |

**Note on output size:** With deduplication enabled, output size may be slightly less than `input_size * multiply` if duplicates were removed. The metadata records actual output size and duplicates removed.

## Future Extensions

1. **LLM-based paraphrasing**: `{"type": "paraphrase", "params": {"style": "formal"}}`
2. **Typo injection**: `{"type": "typo_injection", "params": {"rate": 0.05}}`
3. **Query rephrasing**: `{"type": "query_rephrase"}` to vary user query phrasing
4. **Conditional steps**: Apply step only if previous step was/wasn't applied
5. **Step groups**: Define mutually exclusive steps (e.g., only one language per variant)
6. **Config inheritance**: Base configs that can be extended/overridden
7. **Validation mode**: Dry-run to preview augmentation distribution without generating
8. **DVC integration**: Track datasets with Data Version Control for Git-based versioning
9. **HuggingFace Hub push**: Automatically push augmented datasets to HF Hub with metadata
10. **Lineage visualization**: Generate DAG showing data → augmentation → training → model

## Implementation Plan

### Phase 1: Core Pipeline Infrastructure
1. Define JSON schema for augmentation config
2. Add `--config` argument to CLI
3. Implement pipeline executor with probability-based step application
4. Add metadata columns (`original_index`, `variant_id`, `augmentations`)
5. Implement post-generation deduplication step

### Phase 2: Integrate Existing Features as Pipeline Steps
1. Create `AugmentationStep` base class/protocol
2. Wrap `translate_sample()` as `TranslateStep`
3. Wrap `shuffle_json_functions()` as `ShuffleFunctionsStep`
4. Wrap `randomize_system_instructions_formatting()` as `FormatVariationStep`
5. Wrap `randomize_newline_characters()` as `NewlineVariationStep`
6. Add step registry for dynamic loading

### Phase 3: Metadata & Reproducibility
1. Implement `_augmentation_metadata.json` sidecar file generation
2. Add git commit hash capture (via `subprocess` or `gitpython`)
3. Compute input dataset hash (SHA256 of key columns)
4. Collect augmentation statistics during pipeline execution
5. Add `--track-wandb` flag for W&B artifact logging
6. Link augmentation metadata in finetune command

### Phase 4: Testing & Documentation
1. Add unit tests for pipeline executor
2. Add unit tests for each step type
3. Add unit tests for metadata generation
4. Add integration tests for full augmentation with config
5. Create example config files for common use cases
6. Update CLI help and documentation
7. Add config validation with helpful error messages

### Example Config Files

**`configs/augment_multilingual.json`** - Focus on language diversity:
```json
{
  "multiply": 3,
  "seed": 42,
  "pipeline": [
    {"type": "translate", "probability": 0.8, "params": {"languages": ["fr", "es", "de", "ja", "zh"]}}
  ]
}
```

**`configs/augment_robust.json`** - Focus on format robustness:
```json
{
  "multiply": 2,
  "seed": 42,
  "pipeline": [
    {"type": "shuffle_functions", "probability": 0.9},
    {"type": "format_variation", "probability": 0.7},
    {"type": "newline_variation", "probability": 0.5}
  ]
}
```

**`configs/augment_full.json`** - Maximum diversity:
```json
{
  "multiply": 4,
  "seed": 42,
  "pipeline": [
    {"type": "translate", "probability": 0.5, "params": {"languages": ["fr", "es", "de"]}},
    {"type": "shuffle_functions", "probability": 0.8},
    {"type": "format_variation", "probability": 0.4},
    {"type": "newline_variation", "probability": 0.3}
  ]
}
```

**Note**: These configs omit `input`/`output` to be reusable across datasets:
```bash
# Reuse same config for different datasets
irca dataset augment -c configs/augment_full.json -i datasets/v1 -o datasets/v1-aug
irca dataset augment -c configs/augment_full.json -i datasets/v2 -o datasets/v2-aug
```

## Requirements

This ADR introduces the following requirements (defined in [REQUIREMENTS.md](../REQUIREMENTS.md)):

| Req ID | Name | Summary |
|--------|------|---------|
| FR-DATA-11 | Dataset Size Multiplication | Multiply dataset size by configurable factor |
| FR-DATA-12 | Pipeline Configuration | JSON config with ordered transformation steps |
| FR-DATA-13 | Feature Combination | Multiple features can apply to same variant |
| FR-DATA-14 | Post-Generation Deduplication | Remove duplicate variants |
| FR-DATA-15 | Augmentation Metadata | Sidecar file with git hash, config, stats |
| FR-DATA-16 | W&B Augmentation Tracking | Log as W&B Artifact |
| FR-DATA-17 | Config-Driven CLI | Require JSON config file |

## Acceptance Tests

### FR-DATA-11: Dataset Size Multiplication
```gherkin
Scenario: Dataset size is multiplied by configured factor
  Given a dataset with 100 samples
  And a config with "multiply": 3
  When I run augmentation
  Then the output dataset has approximately 300 samples (minus any deduplicated)
  And each original sample has variant_id 0

Scenario: Original samples are always preserved
  Given a dataset with 10 samples
  And a config with "multiply": 2
  When I run augmentation
  Then all 10 original samples exist with variant_id=0 and augmentations=[]
```

### FR-DATA-12: Pipeline Configuration
```gherkin
Scenario: Config file is validated on load
  Given an invalid config missing "multiply" field
  When I run augmentation
  Then the CLI exits with error "Config validation failed: 'multiply' is required"

Scenario: Pipeline steps are applied in order
  Given a config with pipeline [translate, shuffle_functions]
  When I run augmentation
  Then translation is applied before shuffling for each variant
```

### FR-DATA-13: Feature Combination
```gherkin
Scenario: Multiple features applied to same variant
  Given a config with pipeline steps each at probability 1.0
  When I run augmentation
  Then some variants have multiple augmentations recorded
  And augmentations list contains all applied steps in order

Scenario: Independent probability per step
  Given a config with translate at 0.5 and shuffle at 0.8
  When I run augmentation with seed 42
  Then approximately 50% of variants have translation
  And approximately 80% of variants have shuffle
  And some variants have both (approximately 40%)
```

### FR-DATA-14: Post-Generation Deduplication
```gherkin
Scenario: Duplicate variants are removed
  Given a config with "deduplicate": true
  And a pipeline that can produce duplicates (e.g., only translation with 1 language)
  When I run augmentation with multiply=5
  Then duplicate text variants are removed
  And metadata shows "duplicates_removed" count

Scenario: Deduplication can be disabled
  Given a config with "deduplicate": false
  When I run augmentation
  Then all variants are kept including duplicates
```

### FR-DATA-15: Augmentation Metadata
```gherkin
Scenario: Metadata sidecar file is generated
  Given any valid augmentation config
  When I run augmentation
  Then "_augmentation_metadata.json" exists in output directory
  And it contains "git.commit_hash"
  And it contains "config" (full config content)
  And it contains "statistics.augmentation_counts"
  And it contains "environment.python_version"

Scenario: Metadata includes accurate statistics
  Given a dataset with 100 samples and multiply=2
  When I run augmentation
  Then metadata shows input_samples=100
  And metadata shows actual output count
  And metadata shows per-step augmentation counts
```

### FR-DATA-16: W&B Augmentation Tracking
```gherkin
Scenario: W&B artifact is logged when flag provided
  Given a valid config and W&B credentials configured
  When I run augmentation with --track-wandb
  Then a W&B artifact of type "dataset" is created
  And artifact metadata contains the augmentation config
  And artifact contains the output dataset files

Scenario: W&B tracking is optional
  Given a valid config
  When I run augmentation without --track-wandb
  Then no W&B API calls are made
  And augmentation completes successfully
```

### FR-DATA-17: Config-Driven CLI
```gherkin
Scenario: Config file is required
  Given no --config argument
  When I run "irca dataset augment -i ds -o out"
  Then the CLI exits with error "Missing required option '--config'"

Scenario: CLI paths override config paths
  Given a config with input="datasets/v1" and output="datasets/v1-aug"
  When I run augmentation with --input datasets/v2 --output datasets/v2-aug
  Then input is read from "datasets/v2"
  And output is written to "datasets/v2-aug"

Scenario: Config can omit paths if CLI provides them
  Given a config without input/output fields
  When I run augmentation with --input ds --output out
  Then augmentation succeeds using CLI-provided paths
```

## Links

* Depends on: [ADR-001 Separate Augmentation from Finetuning](./ADR-001-separate-augmentation-finetuning.md)
* Related: [FR-DATA-08 Separated Augmentation Pipeline](../REQUIREMENTS.md)
* Traceability: [TRACEABILITY_MATRIX.md](../TRACEABILITY_MATRIX.md)
