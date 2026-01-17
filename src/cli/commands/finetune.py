"""
Finetune Command

Commands for finetuning models on IRCA datasets.
"""

import logging
import os
import sys

import click

from src.config import get_settings

logger = logging.getLogger(__name__)


@click.group()
def finetune() -> None:
    """Finetune models on IRCA datasets."""
    pass


@finetune.command("run")
@click.option(
    "--model-type",
    "-m",
    type=click.Choice(
        ["mistral", "mistral-v3", "tinyllama", "qwen-7b", "qwen-4b", "qwen-14b", "qwen3-8b", "qwen3-4b", "ministral-3b"]
    ),
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
    help="Dataset to use for training (local path or HuggingFace ID). Must have a 'text' column.",
)
@click.option(
    "--gradient-checkpointing/--no-gradient-checkpointing",
    default=True,
    help="Enable/disable gradient checkpointing (default: enabled)",
)
@click.option(
    "--output",
    "-o",
    type=str,
    default=None,
    help="Output directory for model (overrides default)",
)
@click.option(
    "--batch-size",
    "-bs",
    type=int,
    default=4,
    help="Effective batch size (via gradient accumulation). Default: 4",
)
@click.option(
    "--lora-r",
    type=int,
    default=None,
    help="LoRA rank (overrides config default of 64)",
)
@click.option(
    "--save-steps",
    type=int,
    default=None,
    help="Save checkpoint every N steps (overrides epoch-based saving)",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility. Default: 42",
)
@click.pass_context
def run(
    ctx: click.Context,
    model_type: str,
    epochs: int | None,
    learning_rate: float | None,
    push_to_hub: bool,
    dataset: str | None,
    gradient_checkpointing: bool,
    output: str | None,
    batch_size: int,
    lora_r: int | None,
    save_steps: int | None,
    seed: int,
) -> None:
    """
    Run model finetuning.

    Finetunes a language model using LoRA/QLoRA on a prepared dataset.
    The dataset must have a 'text' column with formatted training prompts.

    For multilingual augmentation, use 'irca dataset augment' first.

    Examples:

        # Finetune Mistral with default settings
        irca finetune run

        # Finetune with a specific dataset
        irca finetune run --dataset ./datasets/my-augmented-dataset

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
    from src.core import utils

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

    # W&B setup
    import wandb

    if settings.wandb_project:
        os.environ["WANDB_PROJECT"] = settings.wandb_project
    if settings.wandb_entity:
        os.environ["WANDB_ENTITY"] = settings.wandb_entity

    try:
        key = settings.wandb_api_key or os.environ.get("WANDB_API_KEY")
        if key:
            wandb.login(key=key)
            click.echo("✅ Logged in to W&B with provided key")
        else:
            wandb.login()
            click.echo("✅ Initialized W&B session (using existing credentials)")
    except Exception as e:
        click.echo(f"⚠️ W&B initialization skip/fail (proceeding anyway): {e}")

    # Set seed for reproducibility
    transformers.set_seed(seed)
    click.echo(f"   Seed: {seed}")

    # Apply CLI overrides
    if epochs is not None:
        config["num_train_epochs"] = epochs
    if learning_rate is not None:
        config["learning_rate"] = learning_rate
    if push_to_hub:
        config["push_to_hub"] = True
    if dataset:
        config["dataset"] = dataset
    if output:
        config["output_dir"] = output
    if lora_r is not None:
        config["lora_r"] = lora_r
    config["gradient_checkpointing"] = gradient_checkpointing
    config["batch_size"] = batch_size
    config["save_steps"] = save_steps

    click.echo(f"   Dataset: {config['dataset']}")
    click.echo(f"   Base model: {config['base_model']}")
    click.echo(f"   Output: {config['output_dir']}")
    click.echo(f"   Epochs: {config['num_train_epochs']}")
    click.echo(f"   Learning rate: {config['learning_rate']}")
    click.echo(f"   LoRA r: {config['lora_r']}, alpha: {config['lora_alpha']}")
    click.echo(f"   Batch size: {batch_size} (effective)")
    if save_steps:
        click.echo(f"   Save strategy: every {save_steps} steps")

    # Login to HuggingFace
    if settings.huggingface_token:
        huggingface_hub.login(token=settings.huggingface_token)
        click.echo("✓ Logged in to HuggingFace Hub")

    # Load dataset
    click.echo(f"\n📊 Loading dataset: {config['dataset']}")
    try:
        dataset_name = config["dataset"].split("/")[-1]
        local_path = settings.datasets_path / dataset_name

        if local_path.exists():
            click.echo(f"   Loading from local disk: {local_path}")
            loaded_dataset = datasets.load_from_disk(str(local_path))
        elif os.path.exists(config["dataset"]):
            click.echo(f"   Loading from direct path: {config['dataset']}")
            loaded_dataset = datasets.load_from_disk(config["dataset"])
        else:
            click.echo("   Attempting to load from HuggingFace Hub...")
            loaded_dataset = datasets.load_dataset(config["dataset"])  # type: ignore

        if isinstance(loaded_dataset, datasets.DatasetDict):
            train_dataset = loaded_dataset["train"]
        else:
            train_dataset = loaded_dataset

        # Validate dataset has required 'text' column
        if "text" not in train_dataset.column_names:
            click.secho(
                f"Error: Dataset must have a 'text' column. Found columns: {train_dataset.column_names}",
                fg="red",
                err=True,
            )
            click.echo("Hint: Use 'irca dataset augment' to prepare your dataset with a 'text' column.")
            sys.exit(1)

        num_examples = len(train_dataset)
        click.echo(f"✓ Dataset loaded: {num_examples} examples")

    except Exception as e:
        click.secho(f"Error loading dataset: {e}", fg="red", err=True)
        sys.exit(1)

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

    # Set up trainer
    click.echo("\n⚙️ Setting up trainer...")

    use_gradient_checkpointing = config.get("gradient_checkpointing", True)

    # Determine save strategy
    if config.get("save_steps"):
        save_strategy = "steps"
        save_steps_value = config["save_steps"]
    else:
        save_strategy = "epoch"
        save_steps_value = None

    # Build training arguments
    training_kwargs = {
        "output_dir": config["output_dir"],
        "num_train_epochs": config["num_train_epochs"],
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": config.get("batch_size", 4),
        "gradient_checkpointing": use_gradient_checkpointing,
        "optim": "adamw_8bit",
        "logging_steps": 2,
        "save_strategy": save_strategy,
        "learning_rate": config["learning_rate"],
        "bf16": True,
        "tf32": True,
        "max_grad_norm": 0.3,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "constant",
        "disable_tqdm": False,
        "max_seq_length": config.get("max_seq_length", 4096),
        "dataset_text_field": "text",
        "packing": False,
        "report_to": "wandb",
        "seed": seed,
    }

    # Add save_steps only if using step-based saving
    if save_steps_value is not None:
        training_kwargs["save_steps"] = save_steps_value

    training_args = trl.SFTConfig(**training_kwargs)

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # Required for gradient checkpointing

    trainer = trl.SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
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
    type=click.Choice(["mistral", "mistral-v3", "tinyllama", "qwen-4b", "qwen-14b", "qwen3-8b", "qwen3-4b"]),
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
