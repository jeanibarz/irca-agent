#!/usr/bin/env python
"""
Test inference script using Unsloth for memory-efficient model loading.

Uses Unsloth's optimized kernels for faster inference with 70% less VRAM.
"""

import json
import os
import sys
from pathlib import Path

# Add src to path FIRST (before any imports from src)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import and set environment variables using centralized constants
from src.core.constants import BASE_TO_UNSLOTH, ENV_TORCHDYNAMO_DISABLE

os.environ[ENV_TORCHDYNAMO_DISABLE] = "1"

# Now import other utilities
from src.diversity.utils import get_lora_base_model, is_lora_adapter, load_model_with_unsloth


def test_inference():
    """Test inference with a finetuned model using Unsloth."""
    # Configuration
    adapter_path = "models/finetuned_models/exp001-baseline"

    # 1. Prepare Prompt
    print("📝 Preparing prompt...")

    system_instructions = "You are a helpful assistant with access to the following functions. Use them if required."

    # Example functions
    functions = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        },
        {
            "name": "calculator",
            "description": "Perform basic arithmetic operations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The mathematical expression to evaluate"}
                },
                "required": ["expression"]
            }
        }
    ]

    # Build prompt using IRCA format
    from src.core.prompt_builder import build_full_prompt

    sample = {
        "system_instructions": system_instructions,
        "example": "",  # No few-shot example for this test
        "available_functions_json": json.dumps(functions, indent=4),
        "user_query": "What's the weather like in Paris today? Also calculate 25 * 4.",
        "assistant_completion": ""  # Empty for generation
    }

    prompt = build_full_prompt(sample)
    print(f"\n--- PROMPT ---\n{prompt}\n--------------")

    # 2. Load Model with Unsloth
    print("\n📦 Loading model with Unsloth (70% less VRAM)...")

    # Detect base model from adapter config
    if is_lora_adapter(adapter_path):
        base_model = get_lora_base_model(adapter_path)
        print(f"   Base model: {base_model}")

        # Get Unsloth-optimized model name
        unsloth_model = BASE_TO_UNSLOTH.get(base_model, base_model)
        print(f"   Unsloth model: {unsloth_model}")
    else:
        raise ValueError(f"Not a LoRA adapter: {adapter_path}")

    model, tokenizer = load_model_with_unsloth(
        model_name=unsloth_model,
        adapter_path=adapter_path,
        max_seq_length=2048,
    )

    print("✓ Model loaded with Unsloth optimizations")

    # 3. Generate
    print("\n🚀 Generating response...")
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the completion part (remove prompt)
    completion = response[len(prompt):] if response.startswith(prompt) else response

    print(f"\n--- RESPONSE ---\n{completion}\n----------------")

    # Cleanup
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test_inference()
