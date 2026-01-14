"""
Model Fine-tuning Script for IRCA-Agent

Fine-tunes a language model using LoRA/QLoRA for function-calling capabilities.

Usage:
    python -m src.finetuning.model_finetuning --model_type mistral
"""

import argparse
import logging
import sys

import datasets
import huggingface_hub
import peft
import torch
import transformers
import trl

from config import get_settings
from core import prompt_builder, utils

# Configure logger
logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.DEBUG) -> logging.Logger:
    """Configure logging for the training process."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(level)
    datasets.utils.logging.set_verbosity(level)
    transformers.utils.logging.set_verbosity(level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    return logger


def setup_peft_config(config: dict) -> peft.LoraConfig:
    """
    Create LoRA configuration for parameter-efficient fine-tuning.

    Based on QLoRA paper recommendations.
    """
    return peft.LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=config["lora_dropout"],
        task_type="CAUSAL_LM",
    )


def load_and_prepare_model(
    config: dict,
    peft_config: peft.LoraConfig,
    settings=None,
) -> tuple:
    """
    Load model and tokenizer, apply quantization and LoRA.

    Args:
        config: Training configuration dictionary
        peft_config: LoRA configuration
        settings: Optional Settings instance

    Returns:
        Tuple of (model, tokenizer)
    """
    settings = settings or get_settings()

    # Login to HuggingFace Hub if token is available
    if settings.huggingface_token:
        huggingface_hub.login(token=settings.huggingface_token)

    model_id = config["base_model"]
    logger.info(f"Loading base model: {model_id}")

    # BitsAndBytesConfig for 4-bit quantization
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.pretraining_tp = 1

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare model for k-bit training and apply LoRA
    model = peft.prepare_model_for_kbit_training(model)
    model = peft.get_peft_model(model, peft_config)

    return model, tokenizer


def setup_trainer(
    model,
    tokenizer,
    dataset,
    config: dict,
    peft_config: peft.LoraConfig | None = None,
) -> trl.SFTTrainer:
    """
    Set up the SFT trainer with training arguments.

    Args:
        model: The model to train
        tokenizer: The tokenizer
        dataset: Training dataset
        config: Training configuration
        peft_config: Optional LoRA configuration

    Returns:
        Configured SFTTrainer instance
    """
    training_args = transformers.TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=10,
        gradient_checkpointing=False,
        optim="adamw_8bit",
        logging_steps=2,
        save_strategy="epoch",
        learning_rate=config["learning_rate"],
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=False,
    )

    return trl.SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        max_seq_length=config.get("max_seq_length", 4096),
        tokenizer=tokenizer,
        packing=True,
        formatting_func=prompt_builder.format_instruction,
        args=training_args,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a model for IRCA Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="mistral",
        choices=["mistral", "tinyllama", "qwen-4b", "qwen-14b"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push trained model to HuggingFace Hub",
    )
    return parser.parse_args()


def main():
    """Main training function."""
    setup_logging()
    logger.info("Starting IRCA-Agent model fine-tuning")

    # Parse arguments
    args = parse_arguments()
    logger.info(f"Model type: {args.model_type}")

    # Load settings from environment
    settings = get_settings()
    logger.debug(f"Workspace directory: {settings.workspace_dir}")

    # Get training configuration
    config = settings.get_training_config(args.model_type)

    # Apply command-line overrides
    if args.epochs is not None:
        config["num_train_epochs"] = args.epochs
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.push_to_hub:
        config["push_to_hub"] = True

    logger.info(f"Training configuration: {config}")

    # Set up LoRA configuration
    peft_config = setup_peft_config(config)
    logger.info("LoRA configuration created")

    # Load model and tokenizer
    logger.info("Loading and preparing model...")
    model, tokenizer = load_and_prepare_model(config, peft_config, settings)
    logger.info("Model loaded and prepared")

    # Print trainable parameters
    utils.print_trainable_parameters(model)

    # Load dataset
    logger.info(f"Loading dataset: {config['dataset']}")
    try:
        dataset = datasets.load_dataset(config["dataset"])
        logger.info(f"Dataset loaded: {len(dataset['train'])} training examples")
    except Exception as e:
        logger.critical(f"Failed to load dataset: {e}")
        sys.exit(1)

    # Set up trainer
    trainer = setup_trainer(model, tokenizer, dataset, config, peft_config)

    # Train
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed")

    # Save model
    logger.info(f"Saving model to: {config['output_dir']}")
    trainer.save_model()
    logger.info(f"Model saved: {config['model_name']}")

    # Push to Hub if requested
    if config.get("push_to_hub"):
        logger.info(f"Pushing model to HuggingFace Hub: {config['model_name']}")
        trainer.model.push_to_hub(config["model_name"])
        logger.info("Model pushed to Hub")

    # Cleanup
    torch.cuda.empty_cache()
    logger.info("Fine-tuning completed successfully")


if __name__ == "__main__":
    main()

