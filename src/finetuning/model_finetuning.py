"""
Model Fine-tuning Script for IRCA-Agent

Fine-tunes a language model using LoRA/QLoRA for function-calling capabilities.

Usage:
    python -m src.finetuning.model_finetuning --model_type mistral
"""

import argparse
import logging
import sys
from typing import Any

import huggingface_hub
import peft
import torch
import transformers
import trl

import datasets  # type: ignore
from config import get_settings
from core import prompt_builder, utils
from datasets import Dataset  # type: ignore

# Configure logger
# Configure logger
logger = logging.getLogger(__name__)


def augment_dataset(original_dataset: Dataset, config: dict[str, Any]) -> Dataset:
    """
    Diversify the dataset by translating a subset of examples.
    Does NOT increase total dataset size (Replacement logic).
    Reasoning traces remain in English.
    """
    if not config.get("augment_enabled"):
        return original_dataset

    import copy
    import random

    from core.generation.constants import FINAL_ANSWER_PROMPT
    from core.prompt_builder import build_full_prompt, parse_corrected_agent_trace
    from dataset_generation.translator import TranslationService

    languages = config["augment_languages"]
    total_ratio = config["augment_ratio"]
    is_dynamic = config.get("augment_dynamic", False)

    logger.info(f"Diversifying dataset (Ratio: {total_ratio}, Languages: {languages}, Dynamic: {is_dynamic})")

    def transform_batch(batch: dict, batch_indices: list[int] | None = None) -> dict:
        # Create a local copy to avoid modifying original if needed
        # But 'batch' is already a copy for map/set_transform

        # Decide which items to translate in THIS batch
        lang_groups: dict[str, list[int]] = {}

        for i in range(len(batch["corrected_agent_trace"])):
            # If we have indices (static mode), we can use them for determinism
            # If not (dynamic/iterable), we use random.random()
            should_diversify = False
            if not is_dynamic and batch_indices is not None:
                # Deterministic based on global index + seed
                random.seed(config.get("augment_seed", 42) + batch_indices[i])
                should_diversify = random.random() < total_ratio
                lang = random.choice(languages) if should_diversify else None
            else:
                # Stochastic (dynamic mode or missing indices)
                should_diversify = random.random() < total_ratio
                lang = random.choice(languages) if should_diversify else None

            if lang:
                if lang not in lang_groups:
                    lang_groups[lang] = []
                lang_groups[lang].append(i)

        if not lang_groups:
            return batch

        # Process each language group
        for lang, local_indices in lang_groups.items():
            translator = TranslationService(
                target_lang=lang,
                model_name_template=config.get("augment_model_name_template", "Helsinki-NLP/opus-mt-en-{lang}"),
                max_length=config.get("augment_max_length", 512),
            )

            # Extract texts
            query_texts = []
            answer_mapping = []  # (local_idx, answer_text_or_None)
            parsed_data = []  # Store parts for reconstruction

            for local_idx in local_indices:
                row_list = batch["corrected_agent_trace"][local_idx]
                full_prompt = row_list[0]["value"].replace("\r\n", "\n")
                parts = parse_corrected_agent_trace(full_prompt)
                parsed_data.append(parts)

                query_texts.append(parts["user_query"])

                comp = parts["assistant_completion"]
                if FINAL_ANSWER_PROMPT in comp:
                    parts_split = comp.rsplit(FINAL_ANSWER_PROMPT, 1)
                    answer_mapping.append(parts_split[1].strip())
                else:
                    answer_mapping.append(None)

            # Translate (using cached model in TranslationService)
            trans_queries = translator.translate_batch(query_texts)

            ans_to_trans = [a for a in answer_mapping if a is not None]
            trans_answers = translator.translate_batch(ans_to_trans) if ans_to_trans else []

            # Reconstruct
            ans_idx = 0
            for i, local_idx in enumerate(local_indices):
                parts = parsed_data[i]
                parts["user_query"] = trans_queries[i]

                if answer_mapping[i] is not None:
                    original_comp = parts["assistant_completion"]
                    pre, _ = original_comp.rsplit(FINAL_ANSWER_PROMPT, 1)
                    parts["assistant_completion"] = f"{pre}{FINAL_ANSWER_PROMPT}{trans_answers[ans_idx]}"
                    ans_idx += 1

                new_full_prompt = build_full_prompt(parts)
                new_row = copy.deepcopy(batch["corrected_agent_trace"][local_idx])
                new_row[0]["value"] = new_full_prompt
                batch["corrected_agent_trace"][local_idx] = new_row

        return batch

    if is_dynamic:
        # Truly online: transform happens every time the item is accessed.
        # Works best with num_workers > 0 in Trainer/DataLoader.
        iterable_ds = original_dataset.to_iterable_dataset()
        return iterable_ds.map(transform_batch, batched=True, batch_size=8)

    # Static (Offline) Diversification
    return original_dataset.map(
        transform_batch, batched=True, batch_size=8, load_from_cache_file=False, desc="Diversifying dataset"
    )


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


def setup_trainer(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    config: dict[str, Any],
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
        dataloader_num_workers=4 if config.get("augment_dynamic") else 0,
        dataloader_pin_memory=True if config.get("augment_dynamic") else False,
    )

    return trl.SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        max_seq_length=config.get("max_seq_length", 4096),
        tokenizer=tokenizer,
        packing=not config.get(
            "augment_dynamic"
        ),  # Disable packing in dynamic mode to allow real-time per-sample variety
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
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable augmentation",
    )
    parser.add_argument(
        "--augment_lang",
        nargs="+",
        help="Languages for augmentation (e.g. fr es)",
    )
    parser.add_argument(
        "--augment_ratio",
        type=float,
        help="Augmentation ratio",
    )
    parser.add_argument(
        "--augment_model_template",
        type=str,
        help="Translation model name template",
    )
    parser.add_argument(
        "--augment_max_length",
        type=int,
        help="Maximum translation length",
    )
    parser.add_argument(
        "--augment_seed",
        type=int,
        help="Random seed for diversification",
    )
    parser.add_argument(
        "--augment_dynamic",
        action="store_true",
        help="Enable online diversification (slower, but varies across runs)",
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
    if args.epochs is not None:
        config["num_train_epochs"] = args.epochs
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.push_to_hub:
        config["push_to_hub"] = True
    if args.augment:
        config["augment_enabled"] = True
    if args.augment_lang:
        config["augment_languages"] = args.augment_lang
    if args.augment_ratio:
        config["augment_ratio"] = args.augment_ratio
    if args.augment_model_template:
        config["augment_model_name_template"] = args.augment_model_template
    if args.augment_max_length:
        config["augment_max_length"] = args.augment_max_length
    if args.augment_seed:
        config["augment_seed"] = args.augment_seed
    if args.augment_dynamic:
        config["augment_dynamic"] = True

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
        dataset = datasets.load_dataset(config["dataset"])  # type: ignore
        logger.info(f"Dataset loaded: {len(dataset['train'])} training examples")

        # Augment dataset
        dataset["train"] = augment_dataset(dataset["train"], config)

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
