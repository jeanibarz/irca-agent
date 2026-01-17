#!/usr/bin/env python
"""
Memory-efficient perplexity evaluation using Unsloth
Uses same memory-efficient loading as training
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_completion_perplexity(model, tokenizer, texts, device="cuda"):
    """Compute perplexity on completion portion only."""
    import torch
    import numpy as np
    from src.diversity.utils import COMPLETION_MARKERS

    perplexities = []

    for text in texts:
        # Find completion start
        completion_start = 0
        for marker in COMPLETION_MARKERS:
            idx = text.find(marker)
            if idx != -1:
                completion_start = idx + len(marker)
                break

        if completion_start == 0:
            # No marker found, use whole text
            prompt = ""
            completion = text
        else:
            prompt = text[:completion_start]
            completion = text[completion_start:]

        if not completion.strip():
            continue

        # Tokenize
        full_text = prompt + completion
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to(device)

        # Get prompt length in tokens
        prompt_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        prompt_len = prompt_tokens["input_ids"].shape[1]

        # Compute loss only on completion tokens
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids, labels=input_ids)

            # Get per-token losses
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Only count completion tokens (after prompt_len - 1 position)
            completion_losses = losses[prompt_len - 1:]

            if len(completion_losses) > 0:
                avg_loss = completion_losses.mean().item()
                ppl = np.exp(avg_loss)
                perplexities.append(ppl)

    return perplexities


def main():
    parser = argparse.ArgumentParser(description="Evaluate with Unsloth (memory-efficient)")
    parser.add_argument("--model", "-m", required=True, help="Path to LoRA adapter")
    parser.add_argument("--dataset", "-d", required=True, help="Path to test dataset")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--base-model", default="unsloth/Qwen3-4B-unsloth-bnb-4bit",
                        help="Base model (default: Unsloth's quantized Qwen3-4B)")
    args = parser.parse_args()

    print(f"Loading model with Unsloth optimizations...")
    print(f"   Adapter: {args.model}")
    print(f"   Base model: {args.base_model}")
    print(f"   Dataset: {args.dataset}")

    # Import unsloth
    from unsloth import FastModel
    import torch
    from datasets import load_from_disk
    import numpy as np

    # Load model with Unsloth's optimizations (memory-efficient)
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=2048,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    # Load LoRA adapter
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.model)
    model.eval()

    print(f"   Model loaded successfully")

    # Check GPU memory
    gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 2)
    print(f"   GPU Memory: {gpu_memory}GB reserved")

    # Load dataset
    dataset = load_from_disk(args.dataset)
    if "text" not in dataset.column_names:
        print(f"Error: Dataset must have 'text' column. Found: {dataset.column_names}")
        sys.exit(1)

    texts = dataset["text"]
    print(f"   Loaded {len(texts)} samples")

    # Compute perplexity
    print(f"\nComputing completion perplexity...")
    perplexities = compute_completion_perplexity(model, tokenizer, texts)

    # Calculate statistics
    results = {
        "model": args.model,
        "dataset": args.dataset,
        "n_samples": len(perplexities),
        "perplexity_mean": float(np.mean(perplexities)),
        "perplexity_std": float(np.std(perplexities)),
        "perplexity_median": float(np.median(perplexities)),
        "perplexity_min": float(np.min(perplexities)),
        "perplexity_max": float(np.max(perplexities)),
    }

    print(f"\nResults:")
    print(f"   Samples: {results['n_samples']}")
    print(f"   Mean perplexity: {results['perplexity_mean']:.4f}")
    print(f"   Std perplexity: {results['perplexity_std']:.4f}")
    print(f"   Median perplexity: {results['perplexity_median']:.4f}")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    else:
        print(f"\n{json.dumps(results, indent=2)}")

    # Cleanup
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
