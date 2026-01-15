"""
Finetune Command

Commands for finetuning models on IRCA datasets.
"""

import logging
import os
import sys

import click

from config import get_settings

logger = logging.getLogger(__name__)


@click.group()
def finetune() -> None:
    """Finetune models on IRCA datasets."""
    pass


@finetune.command("run")
@click.option(
    "--model-type",
    "-m",
    type=click.Choice(["mistral", "mistral-v3", "tinyllama", "qwen-4b", "qwen-14b"]),
    default="mistral",
    help="Type of model to finetune",
)
@click.option(
    "--epochs",
    "-e",
    type=int,
    default=None,
    help="Number of training epochs (overrides config)",
)
@click.option(
    "--learning-rate",
    "-lr",
    type=float,
    default=None,
    help="Learning rate (overrides config)",
)
@click.option(
    "--push-to-hub",
    is_flag=True,
    help="Push trained model to HuggingFace Hub",
)
@click.option(
    "--dataset",
    "-d",
    type=str,
    default=None,
    help="Dataset to use for training (HuggingFace ID)",
)
@click.pass_context
def run(
    ctx: click.Context,
    model_type: str,
    epochs: int | None,
    learning_rate: float | None,
    push_to_hub: bool,
    dataset: str | None,
) -> None:
    """
    Run model finetuning.

    Finetunes a language model using LoRA/QLoRA on the IRCA agent dataset.

    Examples:

        # Finetune Mistral with default settings
        irca finetune run

        # Finetune TinyLlama for 3 epochs
        irca finetune run --model-type tinyllama --epochs 3

        # Finetune and push to HuggingFace Hub
        irca finetune run --push-to-hub
    """
    import huggingface_hub
    import peft
    import torch
    import transformers
    import trl

    import datasets  # type: ignore
    from core import prompt_builder, utils

    settings = get_settings()
    verbose = ctx.obj.get("verbose", False)

    # Set up logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    datasets.utils.logging.set_verbosity(log_level)  # type: ignore
    transformers.utils.logging.set_verbosity(log_level)

    click.echo("🚀 Starting IRCA-Agent finetuning")
    click.echo(f"   Model type: {model_type}")

    # Get training configuration
    config = settings.get_training_config(model_type)

    # Apply CLI overrides
    if epochs is not None:
        config["num_train_epochs"] = epochs
    if learning_rate is not None:
        config["learning_rate"] = learning_rate
    if push_to_hub:
        config["push_to_hub"] = True
    if dataset:
        config["dataset"] = dataset

    click.echo(f"   Dataset: {config['dataset']}")
    click.echo(f"   Base model: {config['base_model']}")
    click.echo(f"   Output: {config['output_dir']}")
    click.echo(f"   Epochs: {config['num_train_epochs']}")
    click.echo(f"   Learning rate: {config['learning_rate']}")
    click.echo(f"   LoRA r: {config['lora_r']}, alpha: {config['lora_alpha']}")

    # Login to HuggingFace
    if settings.huggingface_token:
        huggingface_hub.login(token=settings.huggingface_token)
        click.echo("✓ Logged in to HuggingFace Hub")

    # Set up LoRA config
    click.echo("\n📦 Setting up LoRA configuration...")
    peft_config = peft.LoraConfig(
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

    # Load model
    click.echo(f"\n🔧 Loading model: {config['base_model']}")
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            quantization_config=bnb_config,
            use_cache=False,
            device_map="auto",
            trust_remote_code=True,
        )
        model.config.pretraining_tp = 1

        tokenizer = transformers.AutoTokenizer.from_pretrained(config["base_model"])
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Prepare for training
        model = peft.prepare_model_for_kbit_training(model)
        model = peft.get_peft_model(model, peft_config)
        click.echo("✓ Model loaded and prepared")
    except Exception as e:
        click.secho(f"Error loading model: {e}", fg="red", err=True)
        sys.exit(1)

    # Print trainable parameters
    utils.print_trainable_parameters(model)

    # Load dataset
    click.echo(f"\n📊 Loading dataset: {config['dataset']}")
    try:
        if os.path.exists(config["dataset"]):
            click.echo("   Loading from local disk...")
            loaded_dataset = datasets.load_from_disk(config["dataset"])  # type: ignore
        else:
            loaded_dataset = datasets.load_dataset(config["dataset"])  # type: ignore

        if isinstance(loaded_dataset, datasets.DatasetDict):  # type: ignore
            train_dataset = loaded_dataset["train"]
        else:
            train_dataset = loaded_dataset

        click.echo(f"✓ Dataset loaded: {len(train_dataset)} examples")
    except Exception as e:
        click.secho(f"Error loading dataset: {e}", fg="red", err=True)
        sys.exit(1)

    # Set up trainer
    click.echo("\n⚙️ Setting up trainer...")
    training_args = trl.SFTConfig(
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
        max_seq_length=config.get("max_seq_length", 4096),
        packing=True,
    )

    trainer = trl.SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        formatting_func=prompt_builder.format_instruction,
        args=training_args,
    )

    # Train
    click.echo("\n🏃 Starting training...")
    trainer.train()
    click.echo("✓ Training completed")

    # Save model
    click.echo(f"\n💾 Saving model to: {config['output_dir']}")
    trainer.save_model()
    click.echo(f"✓ Model saved: {config['model_name']}")

    # Push to Hub
    if config.get("push_to_hub"):
        click.echo(f"\n☁️ Pushing to HuggingFace Hub: {config['model_name']}")
        trainer.model.push_to_hub(config["model_name"])
        click.echo("✓ Model pushed to Hub")

    # Cleanup
    torch.cuda.empty_cache()
    click.secho("\n✅ Finetuning completed successfully!", fg="green")


@finetune.command("info")
@click.option(
    "--model-type",
    "-m",
    type=click.Choice(["mistral", "mistral-v3", "tinyllama", "qwen-4b", "qwen-14b"]),
    default="mistral",
    help="Model type to show info for",
)
def info(model_type: str) -> None:
    """
    Show training configuration info.

    Displays the configuration that would be used for finetuning
    without actually starting the training.
    """
    settings = get_settings()
    config = settings.get_training_config(model_type)

    click.echo(f"\n📋 Training Configuration for '{model_type}':\n")
    for key, value in config.items():
        click.echo(f"  {key}: {value}")
    click.echo()
