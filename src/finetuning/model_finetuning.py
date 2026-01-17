"""
Model Fine-tuning Script for IRCA-Agent (Legacy)

DEPRECATED: This module uses the standard transformers backend which requires
~24GB VRAM for 4B models. For 70% less VRAM and 2x faster training, use:

    irca finetune run -m qwen3-4b -d YOUR_DATASET --epochs 3

The CLI uses Unsloth by default for optimized training.

Fine-tunes a language model using LoRA/QLoRA for function-calling capabilities.

Usage (legacy):
    python -m src.finetuning.model_finetuning --model_type mistral
"""

import argparse
import logging
import sys
import warnings
from typing import Any

import huggingface_hub
import peft
import torch
import transformers
import trl

# Emit deprecation warning
warnings.warn(
    "src.finetuning.model_finetuning is deprecated. "
    "Use 'irca finetune run' CLI command instead, which uses Unsloth for "
    "70% less VRAM and 2x faster training.",
    DeprecationWarning,
    stacklevel=2,
)

import datasets  # type: ignore
from src.config import get_settings
from src.core import utils
from src.formatting.chat_template import apply_chat_template, irca_to_messages

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
    datasets.utils.logging.set_verbosity(level)  # type: ignore
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
    config: dict[str, Any],
    peft_config: peft.LoraConfig,
    settings: Any | None = None,
) -> tuple[Any, Any]:
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


def create_formatting_func(tokenizer: Any) -> callable:
    """
    Create a formatting function that applies the model's chat template.

    This ensures training data includes proper EOS tokens for the specific model.

    Args:
        tokenizer: The model's tokenizer.

    Returns:
        A function that formats samples for training.
    """

    def formatting_func(examples: dict[str, Any]) -> list[str]:
        """Format a batch of examples using the model's chat template."""
        texts = []

        # Handle both single example and batched examples
        if isinstance(examples.get("text"), list):
            # Batched - examples is {key: [values]}
            batch_size = len(examples["text"])
            for i in range(batch_size):
                text = examples["text"][i]
                formatted = _format_single_example(text, tokenizer)
                texts.append(formatted)
        else:
            # Single example
            text = examples.get("text", "")
            formatted = _format_single_example(text, tokenizer)
            texts.append(formatted)

        return texts

    return formatting_func


def _format_single_example(text: str, tokenizer: Any) -> str:
    """
    Format a single text example using the model's chat template.

    If the text is already in IRCA format (with ### markers), parse it
    and convert to the model's native chat format.

    Args:
        text: The raw text (potentially in IRCA format).
        tokenizer: The model's tokenizer.

    Returns:
        Formatted text with proper chat template and EOS token.
    """
    from src.core.prompt_builder import parse_corrected_agent_trace

    # Check if this is IRCA format (has our markers)
    if "### ITERATIVE RESOLUTION CYCLE" in text or "### INSTRUCTIONS" in text:
        try:
            # Parse IRCA format to components
            parsed = parse_corrected_agent_trace(text)

            # Convert to messages format
            messages = irca_to_messages(parsed)

            # Apply chat template
            formatted = apply_chat_template(tokenizer, messages, add_generation_prompt=False)

            # Ensure EOS token is present
            if tokenizer.eos_token and not formatted.rstrip().endswith(tokenizer.eos_token):
                formatted = formatted.rstrip() + tokenizer.eos_token

            return formatted
        except Exception as e:
            logger.warning(f"Failed to parse IRCA format, using raw text: {e}")
            # Fall through to raw text handling

    # Not IRCA format or parsing failed - use as-is but ensure EOS
    if tokenizer.eos_token and not text.rstrip().endswith(tokenizer.eos_token):
        text = text.rstrip() + tokenizer.eos_token

    return text


def setup_trainer(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    config: dict[str, Any],
    peft_config: peft.LoraConfig | None = None,
) -> trl.SFTTrainer:
    """
    Set up the SFT trainer with training arguments.

    Uses the model's native chat template for formatting, ensuring proper
    EOS tokens are included in training data.

    Args:
        model: The model to train
        tokenizer: The tokenizer
        dataset: Training dataset (must have 'text' column)
        config: Training configuration
        peft_config: Optional LoRA configuration

    Returns:
        Configured SFTTrainer instance
    """
    # Create formatting function that applies chat template
    formatting_func = create_formatting_func(tokenizer)

    training_args = trl.SFTConfig(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=10,
        gradient_checkpointing=config.get("gradient_checkpointing", True),
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
        max_seq_length=config.get("max_seq_length", 4096),
        packing=False,
        # Note: We use formatting_func instead of dataset_text_field
    )

    logger.info(f"Using chat template formatting for model: {config['base_model']}")
    logger.info(f"EOS token: {tokenizer.eos_token!r} (ID: {tokenizer.eos_token_id})")

    return trl.SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
        formatting_func=formatting_func,  # Apply chat template per sample
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
        choices=[
            "mistral",
            "mistral-v3",
            "tinyllama",
            "qwen-7b",
            "qwen-4b",
            "qwen-14b",
            "qwen3-8b",
            "qwen3-4b",
            "ministral-3b",
        ],
        help="Type of model to train",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset path or HuggingFace ID (must have 'text' column)",
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


def main() -> None:
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
    if args.dataset is not None:
        config["dataset"] = args.dataset
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
        import os

        local_path = settings.datasets_path / config["dataset"].split("/")[-1]

        if os.path.exists(local_path):
            logger.info(f"Loading dataset from local disk: {local_path}")
            loaded_ds = datasets.load_from_disk(str(local_path))
            if isinstance(loaded_ds, datasets.Dataset):
                dataset = datasets.DatasetDict({"train": loaded_ds})
            else:
                dataset = loaded_ds
        elif os.path.exists(config["dataset"]):
            logger.info(f"Loading dataset from path: {config['dataset']}")
            loaded_ds = datasets.load_from_disk(config["dataset"])
            if isinstance(loaded_ds, datasets.Dataset):
                dataset = datasets.DatasetDict({"train": loaded_ds})
            else:
                dataset = loaded_ds
        else:
            logger.info("Dataset not found locally, attempting to load from HuggingFace Hub")
            dataset = datasets.load_dataset(config["dataset"])  # type: ignore

        # Validate dataset has 'text' column
        if "text" not in dataset["train"].column_names:
            logger.critical(f"Dataset must have a 'text' column. Found: {dataset['train'].column_names}")
            sys.exit(1)

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
