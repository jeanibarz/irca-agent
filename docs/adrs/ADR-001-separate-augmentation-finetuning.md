# Separate Data Augmentation from Finetuning Pipeline

* Status: accepted
* Deciders: Jean
* Date: 2026-01-16

Technical Story: Finetuning crashes due to TRL 0.13.0 API incompatibility; complete separation of augmentation simplifies finetuning

## Context and Problem Statement

The finetuning command (`irca finetune run`) is crashing with the error:

```
TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'dataset_text_field'
```

This error occurs because:
1. **TRL library API changed**: The `dataset_text_field` parameter was removed in TRL 0.13.0 (our installed version)
2. **`tokenizer` is also deprecated**: Should use `processing_class` instead

Additionally, the current architecture tightly couples data augmentation (multilingual diversification via translation) with the finetuning process, resulting in:
- Complex CLI with many augmentation-related flags
- Difficult debugging of augmentation vs training issues
- No way to inspect augmented datasets before expensive training
- Mixed responsibilities violating single-responsibility principle

## Decision Drivers

* **Simplicity**: Finetuning command should focus solely on training
* **Separation of concerns**: Augmentation and training are distinct operations
* **Debuggability**: Need to inspect augmented datasets independently
* **Maintainability**: Simpler code is easier to maintain and test
* **API compatibility**: Must adapt to TRL 0.13.0+ API changes

## Decision Outcome

**Complete removal of all augmentation logic from the finetune command.**

The finetune command will:
- Accept only pre-processed datasets with a `text` column
- Have no augmentation-related arguments
- Be simple and focused on training configuration only

Augmentation will be handled by a separate `irca dataset augment` command (future work).

### Positive Consequences

* Drastically simplified finetune command (~50% fewer arguments)
* Clear separation of responsibilities
* Easier to debug and maintain each component
* Faster iteration on training when dataset is already prepared
* No VRAM contention between translation models and training model

### Negative Consequences

* Users must run augmentation as a separate step
* Requires dataset to have pre-formatted `text` column

## Implementation

### Changes to `irca finetune run`

**Removed arguments:**
- `--augment` flag
- `--augment-lang` / `-al`
- `--augment-ratio` / `-ar`
- `--augment-dynamic` / `--no-augment-dynamic`

**Removed code:**
- All augmentation-related imports and logic
- `augment_dataset()` function call
- Dynamic dataset handling (IterableDataset)

**Fixed:**
- TRL 0.13.0 API: `processing_class` instead of `tokenizer`
- TRL 0.13.0 API: removed `dataset_text_field` (handled by SFTConfig)

### Dataset Requirements

The finetune command now expects datasets with:
- A `text` column containing the fully formatted training prompt
- Standard HuggingFace Dataset format (local disk or Hub)

### Future Work

Create `irca dataset augment` command:
```bash
irca dataset augment \
    --input datasets/irca-v1 \
    --output datasets/irca-v1-augmented \
    --languages fr es de \
    --ratio 0.3
```

## Links

* Related: [RFC-003 Multilingual Augmentation](../rfcs/003-multilingual-augmentation.md)
* Related: [TRL Migration Guide](https://github.com/huggingface/trl/releases/tag/v0.13.0)
