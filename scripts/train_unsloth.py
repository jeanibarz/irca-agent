#!/usr/bin/env python
"""
Unsloth-based training script for IRCA-Agent
Uses 70% less VRAM and trains 2x faster than standard HuggingFace/TRL
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Train with Unsloth")
    parser.add_argument("--dataset", "-d", required=True, help="Path to dataset")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--epochs", "-e", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", "-bs", type=int, default=4, help="Effective batch size")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model",
        default="unsloth/Qwen3-4B-unsloth-bnb-4bit",
        help="Model to use (default: Unsloth's optimized Qwen3-4B)",
    )
    args = parser.parse_args()

    print(f"🦥 Starting Unsloth training")
    print(f"   Model: {args.model}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Output: {args.output}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Max seq length: {args.max_seq_length}")
    print(f"   LoRA rank: {args.lora_r}")

    # Import unsloth
    from unsloth import FastModel
    from unsloth import is_bfloat16_supported
    import torch
    from datasets import load_from_disk
    from trl import SFTTrainer, SFTConfig

    # Load model with Unsloth's optimizations
    print("\n📦 Loading model with Unsloth optimizations...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,  # 4-bit quantization for memory efficiency
        load_in_8bit=False,
        full_finetuning=False,
    )

    # Add LoRA adapters
    print("\n🔧 Adding LoRA adapters...")
    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_r,  # Common practice: alpha = r
        lora_dropout=0,  # Unsloth recommends 0 for speed
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=args.seed,
    )

    # Load dataset
    print(f"\n📊 Loading dataset: {args.dataset}")
    dataset = load_from_disk(args.dataset)

    if "text" not in dataset.column_names:
        print(f"Error: Dataset must have 'text' column. Found: {dataset.column_names}")
        sys.exit(1)

    print(f"   Loaded {len(dataset)} examples")

    # Set up training config
    print("\n⚙️ Setting up trainer...")
    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.batch_size,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_steps=5,
        logging_steps=1,
        save_strategy="epoch",
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        optim="adamw_8bit",
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # Show GPU memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"   GPU: {gpu_stats.name}")
    print(f"   GPU Memory: {start_gpu_memory}GB / {max_memory}GB reserved")

    # Train
    print("\n🏃 Starting training...")
    trainer_stats = trainer.train()

    # Show final stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    print(f"\n📈 Training stats:")
    print(f"   Total training time: {trainer_stats.metrics['train_runtime']:.2f}s")
    print(f"   Peak GPU memory: {used_memory}GB")
    print(f"   Memory used for LoRA: {used_memory_for_lora}GB")

    # Save model
    print(f"\n💾 Saving model to: {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("✅ Training completed!")


if __name__ == "__main__":
    main()
