"""
Finetune Command

Commands for finetuning models on IRCA datasets.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import click

from src.config import get_settings

logger = logging.getLogger(__name__)


def _train_with_unsloth(
    settings: "Settings",  # noqa: F821
    config: dict,
    model_type: str,
    train_dataset: "Dataset",  # noqa: F821
    seed: int,
    FastModel,  # noqa: N803 - Matches Unsloth API naming
    is_bfloat16_supported,  # Passed from caller to ensure correct import order
) -> None:
    """
    Train using Unsloth backend (70% less VRAM, 2x faster).

    Uses Unsloth's optimized kernels and pre-quantized 4-bit models.

    Note: FastModel and is_bfloat16_supported are passed as arguments to ensure
    Unsloth is imported BEFORE torch/transformers in the caller.
    """
    import click
    import torch
    import trl

    from src.core import utils

    # Get Unsloth-optimized model name
    try:
        unsloth_model = settings.get_unsloth_model_name(model_type)
    except ValueError as e:
        click.secho(f"Warning: {e}", fg="yellow")
        click.echo("Falling back to standard model with Unsloth's FastModel...")
        unsloth_model = config["base_model"]

    click.echo("\n🦥 Loading model with Unsloth optimizations...")
    click.echo(f"   Unsloth model: {unsloth_model}")

    # Load model with Unsloth's optimizations (with error handling for graceful fallback)
    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=unsloth_model,
            max_seq_length=config.get("max_seq_length", 2048),
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=False,
        )
    except Exception as e:
        # Clean up GPU memory before suggesting fallback
        torch.cuda.empty_cache()
        click.secho(f"\n❌ Unsloth model loading failed: {e}", fg="red", err=True)
        click.echo("\n💡 Suggestions:")
        click.echo("   1. Retry with TRL backend: irca finetune run --backend trl ...")
        click.echo(f"   2. Check if model exists: {unsloth_model}")
        click.echo("   3. Check GPU memory: nvidia-smi")
        raise SystemExit(1) from e

    # Add LoRA adapters
    click.echo("\n📦 Adding LoRA adapters...")
    try:
        model = FastModel.get_peft_model(
            model,
            r=config["lora_r"],
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=config["lora_alpha"],
            lora_dropout=0,  # Unsloth recommends 0 for speed
            bias="none",
            use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
            random_state=seed,
        )
    except Exception as e:
        torch.cuda.empty_cache()
        click.secho(f"\n❌ LoRA adapter setup failed: {e}", fg="red", err=True)
        click.echo("\n💡 Suggestions:")
        click.echo("   1. Retry with TRL backend: irca finetune run --backend trl ...")
        click.echo("   2. Check LoRA parameters (lora_r, lora_alpha)")
        raise SystemExit(1) from e

    click.echo("✓ Model loaded and prepared")
    utils.print_trainable_parameters(model)

    # Show GPU memory stats
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        click.echo(f"   GPU: {gpu_stats.name}")
        click.echo(f"   GPU Memory: {start_gpu_memory}GB / {max_memory}GB reserved")

    # Set up trainer
    click.echo("\n⚙️ Setting up trainer...")

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
        "learning_rate": config["learning_rate"],
        "lr_scheduler_type": "linear",
        "warmup_steps": 5,
        "logging_steps": 2,
        "save_strategy": save_strategy,
        "bf16": is_bfloat16_supported(),
        "fp16": not is_bfloat16_supported(),
        "optim": "adamw_8bit",
        "seed": seed,
        "max_seq_length": config.get("max_seq_length", 2048),
        "dataset_text_field": "text",
        "packing": False,
        "report_to": "wandb" if config.get("wandb_enabled", False) else "none",
    }

    if save_steps_value is not None:
        training_kwargs["save_steps"] = save_steps_value

    training_args = trl.SFTConfig(**training_kwargs)

    trainer = trl.SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )

    # Train
    click.echo("\n🏃 Starting training...")
    trainer_stats = trainer.train()
    click.echo("✓ Training completed")

    # Finalize W&B run with final metrics
    import wandb

    if wandb.run is not None:
        wandb.log(
            {
                "final/train_loss": trainer_stats.metrics.get("train_loss"),
                "final/train_runtime": trainer_stats.metrics.get("train_runtime"),
                "final/samples_per_second": trainer_stats.metrics.get("train_samples_per_second"),
            }
        )
        wandb.finish()
        click.echo("✓ W&B run finalized")

    # Show final stats
    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        click.echo("\n📈 Training stats:")
        click.echo(f"   Total training time: {trainer_stats.metrics['train_runtime']:.2f}s")
        click.echo(f"   Peak GPU memory: {used_memory}GB")

    # Save model
    click.echo(f"\n💾 Saving model to: {config['output_dir']}")
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    click.echo(f"✓ Model saved: {config['model_name']}")

    # Push to Hub
    if config.get("push_to_hub"):
        click.echo(f"\n☁️ Pushing to HuggingFace Hub: {config['model_name']}")
        model.push_to_hub(config["model_name"])
        click.echo("✓ Model pushed to Hub")


def _train_with_trl(
    config: dict,
    train_dataset: "Dataset",  # noqa: F821
    seed: int,
    peft_module,
    transformers_module,
    trl_module,
    torch_module,
) -> None:
    """
    Train using standard TRL backend with BitsAndBytes quantization.

    Requires more VRAM (~24GB for 4B models) but works without Unsloth.
    """
    import click

    from src.core import utils

    # Set up LoRA config
    click.echo("\n📦 Setting up LoRA configuration...")
    peft_config = peft_module.LoraConfig(
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
    bnb_config = transformers_module.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_module.bfloat16,
    )

    try:
        model = transformers_module.AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            quantization_config=bnb_config,
            use_cache=False,
            device_map="auto",
            trust_remote_code=True,
        )
        model.config.pretraining_tp = 1

        tokenizer = transformers_module.AutoTokenizer.from_pretrained(config["base_model"])
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Prepare for training
        model = peft_module.prepare_model_for_kbit_training(model)
        model = peft_module.get_peft_model(model, peft_config)
        click.echo("✓ Model loaded and prepared")
    except Exception as e:
        click.secho(f"Error loading model: {e}", fg="red", err=True)
        raise SystemExit(1) from e

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
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
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
        "report_to": "wandb" if config.get("wandb_enabled", False) else "none",
        "seed": seed,
    }

    if save_steps_value is not None:
        training_kwargs["save_steps"] = save_steps_value

    training_args = trl_module.SFTConfig(**training_kwargs)

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    trainer = trl_module.SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    # Train
    click.echo("\n🏃 Starting training...")
    trainer_stats = trainer.train()
    click.echo("✓ Training completed")

    # Finalize W&B run with final metrics
    import wandb

    if wandb.run is not None:
        wandb.log(
            {
                "final/train_loss": trainer_stats.metrics.get("train_loss"),
                "final/train_runtime": trainer_stats.metrics.get("train_runtime"),
                "final/samples_per_second": trainer_stats.metrics.get("train_samples_per_second"),
            }
        )
        wandb.finish()
        click.echo("✓ W&B run finalized")

    # Save model
    click.echo(f"\n💾 Saving model to: {config['output_dir']}")
    trainer.save_model()
    click.echo(f"✓ Model saved: {config['model_name']}")

    # Push to Hub
    if config.get("push_to_hub"):
        click.echo(f"\n☁️ Pushing to HuggingFace Hub: {config['model_name']}")
        trainer.model.push_to_hub(config["model_name"])
        click.echo("✓ Model pushed to Hub")


@click.group()
def finetune() -> None:
    """Finetune models on IRCA datasets."""
    pass


@finetune.command("run")
@click.option(
    "--model-type",
    "-m",
    type=click.Choice(["mistral", "mistral-v3", "tinyllama", "qwen-7b", "qwen-4b", "qwen-14b", "qwen3-8b", "qwen3-4b"]),
    default="qwen3-4b",
    help="Type of model to finetune",
)
@click.option(
    "--backend",
    type=click.Choice(["unsloth", "trl"]),
    default="unsloth",
    help="Training backend: unsloth (70%% less VRAM, 2x faster) or trl (standard)",
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
    help="LoRA rank (default: 16, use higher for more capacity but more VRAM)",
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
@click.option(
    "--max-seq-length",
    type=int,
    default=1024,
    help="Maximum sequence length for training. Default: 1024 (memory efficient)",
)
@click.option(
    "--no-wandb",
    is_flag=True,
    help="Disable W&B logging even if configured",
)
@click.pass_context
def run(
    ctx: click.Context,
    model_type: str,
    backend: str,
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
    max_seq_length: int,
    no_wandb: bool,
) -> None:
    """
    Run model finetuning.

    Finetunes a language model using LoRA/QLoRA on a prepared dataset.
    The dataset must have a 'text' column with formatted training prompts.

    Two backends are available:

    - unsloth (default): Uses Unsloth's optimized kernels for 70% less VRAM
      and 2x faster training. Requires ~6GB VRAM for 4B models.

    - trl: Standard TRL/BitsAndBytes training. Requires ~24GB VRAM for 4B models.

    Examples:

        # Finetune with Unsloth (default, memory-efficient)
        irca finetune run -m qwen3-4b -d ./datasets/my-dataset --epochs 3

        # Finetune with TRL backend (if Unsloth unavailable)
        irca finetune run -m qwen3-4b -d ./datasets/my-dataset --backend trl

        # Finetune and push to HuggingFace Hub
        irca finetune run --push-to-hub
    """
    # =========================================================================
    # CRITICAL: Import order matters for Unsloth!
    # Unsloth MUST be imported BEFORE torch/transformers to apply optimizations.
    # =========================================================================
    from src.core.constants import (
        ENV_TORCHDYNAMO_DISABLE,
        get_unsloth_availability,
    )

    # Set environment variables BEFORE any ML imports
    os.environ[ENV_TORCHDYNAMO_DISABLE] = "1"

    # Import Unsloth FIRST if using unsloth backend
    FastModel = None  # noqa: N806 - Matches Unsloth API naming
    is_bfloat16_supported = None
    if backend == "unsloth":
        available, reason = get_unsloth_availability()
        if not available:
            click.secho(
                f"Error: Unsloth not available: {reason}",
                fg="red",
                err=True,
            )
            click.echo("Tip: Use --backend trl for standard training without Unsloth.")
            raise SystemExit(1)

        from unsloth import FastModel, is_bfloat16_supported  # noqa: N811

    # NOW import other ML libraries (after Unsloth)
    import huggingface_hub
    import peft
    import torch
    import transformers
    import trl

    import datasets  # type: ignore

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

    backend_emoji = "🦥" if backend == "unsloth" else "🚀"
    click.echo(f"{backend_emoji} Starting IRCA-Agent finetuning ({backend} backend)")
    click.echo(f"   Model type: {model_type}")
    click.echo(f"   Backend: {backend}")

    # Get training configuration
    config = settings.get_training_config(model_type)

    # W&B setup - login first, init later after config is complete
    import wandb

    wandb_enabled = not no_wandb and settings.wandb_project
    if wandb_enabled:
        if settings.wandb_project:
            os.environ["WANDB_PROJECT"] = settings.wandb_project
        if settings.wandb_entity:
            os.environ["WANDB_ENTITY"] = settings.wandb_entity

        try:
            key = settings.wandb_api_key or os.environ.get("WANDB_API_KEY")
            if key:
                wandb.login(key=key)
                click.echo("✅ Logged in to W&B")
            else:
                wandb.login()
                click.echo("✅ W&B session ready")
        except Exception as e:
            click.echo(f"⚠️ W&B login failed (proceeding without tracking): {e}")
            wandb_enabled = False
    elif no_wandb:
        click.echo("   W&B logging disabled (--no-wandb flag)")

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
    config["max_seq_length"] = max_seq_length

    click.echo(f"   Dataset: {config['dataset']}")
    click.echo(f"   Base model: {config['base_model']}")
    click.echo(f"   Output: {config['output_dir']}")
    click.echo(f"   Epochs: {config['num_train_epochs']}")
    click.echo(f"   Learning rate: {config['learning_rate']}")
    click.echo(f"   LoRA r: {config['lora_r']}, alpha: {config['lora_alpha']}")
    click.echo(f"   Batch size: {batch_size} (effective)")
    click.echo(f"   Max seq length: {max_seq_length}")
    if save_steps:
        click.echo(f"   Save strategy: every {save_steps} steps")

    # Store wandb_enabled in config for training functions
    config["wandb_enabled"] = wandb_enabled

    # Initialize W&B run BEFORE training (explicit init for reliable tracking)
    if wandb_enabled:
        try:
            dataset_stem = Path(config["dataset"]).stem
            run_name = f"{model_type}-{dataset_stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project=settings.wandb_project,
                entity=settings.wandb_entity,
                name=run_name,
                job_type="finetuning",
                config={
                    "model_type": model_type,
                    "backend": backend,
                    "base_model": config["base_model"],
                    "dataset": config["dataset"],
                    "epochs": config["num_train_epochs"],
                    "learning_rate": config["learning_rate"],
                    "lora_r": config["lora_r"],
                    "lora_alpha": config["lora_alpha"],
                    "batch_size": batch_size,
                    "max_seq_length": max_seq_length,
                    "seed": seed,
                    "save_steps": save_steps,
                },
                reinit=True,  # Allow re-initialization if needed
            )
            click.echo(f"📊 W&B run initialized: {run_name}")
        except Exception as e:
            click.echo(f"⚠️ W&B init failed (continuing without tracking): {e}")
            wandb_enabled = False
            config["wandb_enabled"] = False

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

    # =========================================================================
    # BACKEND DISPATCH: Unsloth vs TRL
    # =========================================================================
    if backend == "unsloth":
        _train_with_unsloth(
            settings=settings,
            config=config,
            model_type=model_type,
            train_dataset=train_dataset,
            seed=seed,
            FastModel=FastModel,
            is_bfloat16_supported=is_bfloat16_supported,
        )
    else:
        _train_with_trl(
            config=config,
            train_dataset=train_dataset,
            seed=seed,
            peft_module=peft,
            transformers_module=transformers,
            trl_module=trl,
            torch_module=torch,
        )

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
