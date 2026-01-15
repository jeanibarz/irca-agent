# Finetuning

This document explains how to finetune models using the generated datasets.

## Overview

`irca-agent` uses **Q-LoRA** (Quantized Low-Rank Adaptation) to finetune large language models efficiently on consumer hardware. We leverage the HuggingFace `transformers`, `peft`, and `bitsandbytes` libraries.

## Supported Models

The system is pre-configured to support several base models:

- **Mistral**: `mistralai/Mistral-7B-Instruct-v0.2` (Default)
- **TinyLlama**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (Fast, good for debugging)
- **Qwen**: `Qwen/Qwen2.5-3B-Instruct` and 14B variants.

You can extend support to other models by updating `src/config/settings.py`.

## Configuration

Training parameters are controlled via `src/config/settings.py` or environment variables.

### Key Hyperparameters (`config.Settings`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 128 | Rank of the LoRA matrices. Higher = more parameters. |
| `lora_alpha` | 64 | Scaling factor for LoRA updates. |
| `lora_dropout` | 0.05 | Dropout probability for LoRA layers. |
| `learning_rate` | 1e-3 | Initial learning rate. |
| `num_train_epochs` | 5 | Number of passes through the dataset. |
| `max_seq_length` | 4096 | Maximum context length. |

## Running Finetuning

### 1. Prepare Dataset
Ensure you have a generated dataset available locally or on HuggingFace Hub. The finetuner expects data in the format produced by `irca generate traces`.

### 2. Execute Command

```bash
# Standard run (Mistral)
irca finetune run

# Select specific model
irca finetune run --model-type tinyllama

# Override hyperparameters
export LEARNING_RATE=2e-4
irca finetune run
```

### 3. Monitoring
Training logs (loss, learning rate) are printed to the console. Optionally, you can enable Weights & Biases logging by setting `WANDB_API_KEY`.

## Artifacts

After training, the adapter weights are saved to `finetuned_models/<model_name>_<timestamp>`.
The artifacts include:
- `adapter_config.json`: LoRA configuration.
- `adapter_model.bin`: Trained weights.
- `tokenizer.json`: Tokenizer files.

## Multilingual Data Augmentation

To improve model robustness and enable it to handle queries in different languages while maintaining English-based reasoning, `irca-agent` supports on-the-fly dataset augmentation.

### How it Works
The system diversifies a configurable subset of the training dataset's `User Queries` and `Final Answers` by translating them into target languages (e.g., French, Spanish) during the data loading phase. Reasoning traces remain in English to maintain internal logic consistency. This is implemented using **replacement logic**, meaning the total dataset size remains constant while its linguistic diversity increases.

### Configuration

| CLI Flag | Environment Variable | Default | Description |
|----------|----------------------|---------|-------------|
| `--augment` | `AUGMENT_ENABLED` | `False` | Enable multilingual augmentation. |
| `--augment_lang` | `AUGMENT_LANGUAGES` | `['fr']` | List of target languages (space separated in CLI). |
| `--augment_ratio` | `AUGMENT_RATIO` | `0.5` | Probability (ratio) of a sample being augmented. |
| `--augment_model_template` | `AUGMENT_MODEL_NAME_TEMPLATE` | `Helsinki-NLP/opus-mt-en-{lang}` | HF template for translation models. |
| `--augment_max_length` | `AUGMENT_MAX_LENGTH` | `512` | Max tokens for translation. |

### Example

```bash
# Augment 20% of the dataset with French and Spanish translations
irca finetune run --augment --augment_lang fr es --augment_ratio 0.2
```

## Inference with Finetuned Models

To use your finetuned model:

1. Load the base model.
2. Load the LoRA adapter using `PeftModel`.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = PeftModel.from_pretrained(base_model, "finetuned_models/my_run")
```
