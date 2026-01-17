# Finetuning

This document explains how to finetune models using the generated datasets.

## Overview

`irca-agent` uses **Q-LoRA** (Quantized Low-Rank Adaptation) to finetune large language models efficiently on consumer hardware.

### Backend Options

Two backends are available:

| Backend | VRAM Usage | Speed | Default |
|---------|------------|-------|---------|
| **Unsloth** | ~6GB for 4B models | 2x faster | Yes |
| **TRL** | ~24GB for 4B models | Standard | No |

**Unsloth** (default) uses optimized kernels and pre-quantized 4-bit models for significantly reduced memory usage and faster training.

## Supported Models

The system is pre-configured to support several base models with Unsloth-optimized versions:

| Model Type | HuggingFace Model | Unsloth Model |
|------------|-------------------|---------------|
| `qwen3-4b` | `Qwen/Qwen3-4B` | `unsloth/Qwen3-4B-unsloth-bnb-4bit` |
| `qwen3-8b` | `Qwen/Qwen3-8B` | `unsloth/Qwen3-8B-unsloth-bnb-4bit` |
| `mistral` | `mistralai/Mistral-7B-Instruct-v0.2` | `unsloth/mistral-7b-instruct-v0.2-bnb-4bit` |
| `mistral-v3` | `mistralai/Mistral-7B-Instruct-v0.3` | `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` |
| `tinyllama` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `unsloth/tinyllama-bnb-4bit` |

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
Ensure you have a generated dataset available locally or on HuggingFace Hub. The dataset must have a `text` column with formatted training prompts.

```bash
# Format dataset with augmentation
irca dataset format \
  -i datasets/my-raw-dataset \
  -o datasets/my-formatted-dataset \
  -c configs/format_full_augment.json
```

### 2. Execute Training

```bash
# Default: Unsloth backend (recommended, 70% less VRAM)
irca finetune run -m qwen3-4b -d datasets/my-formatted-dataset --epochs 3

# TRL backend (if Unsloth unavailable)
irca finetune run -m qwen3-4b -d datasets/my-formatted-dataset --epochs 3 --backend trl

# Override hyperparameters
irca finetune run -m qwen3-4b -d datasets/my-formatted-dataset --epochs 3 --learning-rate 2e-4

# Save checkpoints every N steps
irca finetune run -m qwen3-4b -d datasets/my-formatted-dataset --save-steps 50
```

### 3. Monitoring
Training logs (loss, learning rate) are printed to the console. W&B logging is enabled by default if `WANDB_API_KEY` is set.

## Artifacts

After training, the adapter weights are saved to `models/finetuned_models/<model_name>_<dataset>_<timestamp>`.
The artifacts include:
- `adapter_config.json`: LoRA configuration.
- `adapter_model.safetensors`: Trained weights.
- `tokenizer.json`: Tokenizer files.

## Inference with Finetuned Models

### Option 1: Use the Playground (Recommended)

The IRCA playground uses Unsloth for inference, providing the same memory efficiency as training.

```bash
# Start the playground
irca playground start

# Check status (shows inference_backend: unsloth)
curl http://localhost:8000/health

# Load your model via the UI or API
```

### Option 2: Direct Python (Unsloth)

```python
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from src.diversity.utils import load_model_with_unsloth

# Load model with Unsloth (70% less VRAM)
model, tokenizer = load_model_with_unsloth(
    model_name="unsloth/Qwen3-4B-unsloth-bnb-4bit",
    adapter_path="models/finetuned_models/my-adapter",
)

# Generate
inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### Option 3: Direct Python (Standard transformers)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Load base model with quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
    quantization_config=bnb_config,
    device_map="auto",
)

# Load adapter
model = PeftModel.from_pretrained(base_model, "models/finetuned_models/my-adapter")
model.eval()
```

## Disabling Unsloth

To force the standard transformers backend:

```bash
# For training
irca finetune run --backend trl ...

# For inference (playground)
export IRCA_DISABLE_UNSLOTH=1
irca playground start
```

## Multilingual Data Augmentation

To improve model robustness and enable it to handle queries in different languages while maintaining English-based reasoning, use the dataset augmentation features.

### Configuration

See `configs/format_full_augment.json` for the full augmentation configuration:

```json
{
  "multiply": 3,
  "structural": [
    {
      "type": "translate",
      "probability": 0.67,
      "params": {
        "languages": ["fr", "es"],
        "targets": ["query", "answer"]
      }
    }
  ],
  "formatting": [
    {"type": "shuffle_functions", "probability": 0.8},
    {"type": "indent_functions", "probability": 0.5}
  ]
}
```

### Example

```bash
# Format dataset with multilingual augmentation
irca dataset format \
  -i datasets/irca_agent_dataset_v5-5acc \
  -o datasets/irca_agent_dataset_v5-5acc-formatted-augmented \
  -c configs/format_full_augment.json
```
