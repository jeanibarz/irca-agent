
import json
import sys
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add src to path to import core modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.prompt_builder import build_full_prompt

def test_inference():
    # Configuration
    base_model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    adapter_path = "models/finetuned_models/Mistral-7B-Instruct-v0.3_irca_agent_v5-6"

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

    sample = {
        "system_instructions": system_instructions,
        "example": "", # No few-shot example for this test
        "available_functions_json": json.dumps(functions, indent=4),
        "user_query": "What's the weather like in Paris today? Also calculate 25 * 4.",
        "assistant_completion": "" # Empty for generation
    }

    prompt = build_full_prompt(sample)
    print(f"\n--- PROMPT ---\n{prompt}\n--------------")

    # 2. Load Model
    print("\n📦 Loading model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # 3. Generate
    print("\n🚀 Generating response...")
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

if __name__ == "__main__":
    test_inference()
