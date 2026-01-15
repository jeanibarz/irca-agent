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
