"""
Dataset Command

Commands for managing datasets (Argilla, HuggingFace Hub).
"""

import logging
import sys

import click

from src.config import get_settings

logger = logging.getLogger(__name__)


@click.group()
def dataset() -> None:
    """Manage datasets (Argilla, HuggingFace Hub)."""
    pass


@dataset.command("push")
@click.option(
    "--source",
    "-s",
    type=str,
    required=True,
    help="Source dataset (Argilla name or local path)",
)
@click.option(
    "--target",
    "-t",
    type=str,
    required=True,
    help="Target HuggingFace dataset ID (e.g., username/dataset-name)",
)
@click.option(
    "--workspace",
    "-w",
    type=str,
    default="irca_agent",
    help="Argilla workspace name",
)
@click.pass_context
def push(
    ctx: click.Context,
    source: str,
    target: str,
    workspace: str,
) -> None:
    """
    Push dataset to HuggingFace Hub.

    Exports a dataset from Argilla and pushes it to HuggingFace Hub.

    Examples:

        # Push Argilla dataset to Hub
        irca dataset push --source irca_agent_dataset_v5-5 --target JeanIbarz/my-dataset
    """
    settings = get_settings()
    verbose = ctx.obj.get("verbose", False)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        import argilla as rg

        # Initialize Argilla
        click.echo(f"Connecting to Argilla at {settings.argilla_api_url}...")
        rg.init(
            api_url=settings.argilla_api_url,
            api_key=settings.argilla_api_key,
        )

        # Load dataset
        click.echo(f"Loading dataset '{source}' from workspace '{workspace}'...")
        ds = rg.FeedbackDataset.from_argilla(name=source, workspace=workspace)

        # Filter submitted records
        records = [r for r in ds.records if r.status == "submitted"]
        click.echo(f"Found {len(records)} submitted records")

        if not records:
            click.secho("No submitted records to push!", fg="yellow")
            return

        # Push to Hub
        click.echo(f"Pushing to HuggingFace Hub: {target}...")
        ds.push_to_huggingface(target)

        click.secho(f"✅ Successfully pushed to {target}", fg="green")

    except ImportError:
        click.secho("Error: Argilla not installed. Install with: pip install argilla", fg="red", err=True)
        sys.exit(1)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


@dataset.command("pull")
@click.option(
    "--source",
    "-s",
    type=str,
    required=True,
    help="HuggingFace dataset ID",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Local output directory",
)
@click.pass_context
def pull(
    ctx: click.Context,
    source: str,
    output: str | None,
) -> None:
    """
    Pull dataset from HuggingFace Hub.

    Downloads a dataset from HuggingFace Hub to local storage.

    Examples:

        # Pull dataset to default location
        irca dataset pull --source JeanIbarz/irca_agent_dataset_v5-5acc

        # Pull to specific directory
        irca dataset pull --source JeanIbarz/irca_agent_dataset_v5-5acc --output ./my-data
    """
    from pathlib import Path

    from datasets import load_dataset  # type: ignore

    settings = get_settings()
    verbose = ctx.obj.get("verbose", False)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    click.echo(f"Downloading dataset: {source}...")

    try:
        ds = load_dataset(source)
        click.echo(f"✓ Downloaded {len(ds['train'])} training examples")

        # Save to disk if output specified
        if output:
            output_path = Path(output)
        else:
            dataset_name = source.split("/")[-1]
            output_path = settings.datasets_path / dataset_name

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(output_path))
        click.secho(f"✅ Saved to {output_path}", fg="green")

    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


@dataset.command("list")
@click.option(
    "--workspace",
    "-w",
    type=str,
    default="irca_agent",
    help="Argilla workspace to list",
)
@click.pass_context
def list_datasets(ctx: click.Context, workspace: str) -> None:
    """
    List datasets in Argilla workspace.

    Shows all available datasets in the specified Argilla workspace.
    """
    settings = get_settings()

    try:
        import argilla as rg

        click.echo(f"Connecting to Argilla at {settings.argilla_api_url}...")
        rg.init(
            api_url=settings.argilla_api_url,
            api_key=settings.argilla_api_key,
        )

        click.echo(f"\n📋 Datasets in workspace '{workspace}':\n")

        # List datasets
        datasets_list = rg.FeedbackDataset.list(workspace=workspace)
        if not datasets_list:
            click.echo("  No datasets found")
            return

        for ds in datasets_list:
            click.echo(f"  • {ds.name}")

    except ImportError:
        click.secho("Error: Argilla not installed", fg="red", err=True)
        sys.exit(1)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


@dataset.command("inspect")
@click.option(
    "--path",
    "-p",
    type=str,
    required=True,
    help="Dataset path (local or HuggingFace ID)",
)
@click.option(
    "--samples",
    "-n",
    type=int,
    default=3,
    help="Number of samples to display",
)
@click.pass_context
def inspect(ctx: click.Context, path: str, samples: int) -> None:
    """
    Inspect dataset structure and sample contents.

    Shows column names, dataset size, and sample entries to verify
    the dataset is correctly formatted for finetuning.

    Examples:

        # Inspect local dataset
        irca dataset inspect -p ./datasets/my-dataset

        # Inspect with more samples
        irca dataset inspect -p ./datasets/my-dataset -n 5
    """
    import os

    from datasets import load_dataset, load_from_disk  # type: ignore

    settings = get_settings()

    click.echo(f"🔍 Inspecting dataset: {path}\n")

    try:
        dataset_name = path.split("/")[-1]
        local_path = settings.datasets_path / dataset_name

        if local_path.exists():
            loaded_dataset = load_from_disk(str(local_path))
        elif os.path.exists(path):
            loaded_dataset = load_from_disk(path)
        else:
            loaded_dataset = load_dataset(path)

        if hasattr(loaded_dataset, "keys") and "train" in loaded_dataset:
            ds = loaded_dataset["train"]
            click.echo("📁 Type: DatasetDict with 'train' split")
        else:
            ds = loaded_dataset
            click.echo("📁 Type: Dataset")

        click.echo(f"📊 Size: {len(ds)} examples")
        click.echo(f"📋 Columns: {ds.column_names}")

        # Check for required 'text' column
        if "text" in ds.column_names:
            click.secho("✓ Has 'text' column (ready for finetuning)", fg="green")
        else:
            click.secho("✗ Missing 'text' column (needs augmentation)", fg="yellow")

        # Show sample entries
        click.echo(f"\n📝 Sample entries ({min(samples, len(ds))} of {len(ds)}):\n")

        for i in range(min(samples, len(ds))):
            click.echo(f"{'=' * 60}")
            click.echo(f"Sample {i + 1}:")
            click.echo(f"{'=' * 60}")

            sample = ds[i]
            for col in ds.column_names:
                value = sample[col]
                if isinstance(value, str):
                    # Truncate long strings
                    if len(value) > 500:
                        display_value = value[:500] + "..."
                    else:
                        display_value = value
                    click.echo(f"\n{col}:")
                    click.echo(f"  {display_value}")
                elif isinstance(value, list) and len(value) > 0:
                    click.echo(f"\n{col}: (list with {len(value)} items)")
                    if isinstance(value[0], dict):
                        # Show first item if it's a dict
                        first_val = str(value[0])
                        if len(first_val) > 300:
                            first_val = first_val[:300] + "..."
                        click.echo(f"  First item: {first_val}")
                else:
                    click.echo(f"\n{col}: {value}")

    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


@dataset.command("augment")
@click.option(
    "--input",
    "-i",
    "input_path",
    type=str,
    required=True,
    help="Input dataset (local path or HuggingFace ID)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output path for augmented dataset",
)
@click.option(
    "--languages",
    "-l",
    multiple=True,
    required=True,
    help="Target languages for translation (e.g., -l fr -l es)",
)
@click.option(
    "--ratio",
    "-r",
    type=float,
    default=0.5,
    help="Probability of translating each sample (0.0 to 1.0)",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility",
)
@click.pass_context
def augment(
    ctx: click.Context,
    input_path: str,
    output: str,
    languages: tuple[str, ...],
    ratio: float,
    seed: int,
) -> None:
    """
    Augment dataset with multilingual translations.

    Translates a portion of the dataset to specified languages while
    keeping reasoning traces in English. Only user queries and final
    answers are translated.

    Examples:

        # Augment with French and Spanish, 50% probability
        irca dataset augment -i ./datasets/irca-v1 -o ./datasets/irca-v1-augmented -l fr -l es -r 0.5

        # Augment with German only, 30% probability
        irca dataset augment -i JeanIbarz/irca-dataset -o ./augmented -l de -r 0.3
    """
    import os
    import random
    from pathlib import Path

    import torch

    from datasets import load_dataset, load_from_disk  # type: ignore
    from src.core.generation.constants import FINAL_ANSWER_PROMPT
    from src.core.prompt_builder import build_full_prompt, parse_corrected_agent_trace
    from src.dataset_generation.translator import TranslationService

    settings = get_settings()
    verbose = ctx.obj.get("verbose", False)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    click.echo("🌍 Dataset Augmentation")
    click.echo(f"   Input: {input_path}")
    click.echo(f"   Output: {output}")
    click.echo(f"   Languages: {', '.join(languages)}")
    click.echo(f"   Ratio: {ratio:.0%}")
    click.echo(f"   Seed: {seed}")

    # Load dataset
    click.echo("\n📊 Loading dataset...")
    try:
        dataset_name = input_path.split("/")[-1]
        local_path = settings.datasets_path / dataset_name

        if local_path.exists():
            click.echo(f"   Loading from: {local_path}")
            loaded_dataset = load_from_disk(str(local_path))
        elif os.path.exists(input_path):
            click.echo(f"   Loading from: {input_path}")
            loaded_dataset = load_from_disk(input_path)
        else:
            click.echo(f"   Loading from HuggingFace Hub: {input_path}")
            loaded_dataset = load_dataset(input_path)

        if hasattr(loaded_dataset, "keys") and "train" in loaded_dataset:
            original_dataset = loaded_dataset["train"]
        else:
            original_dataset = loaded_dataset

        num_examples = len(original_dataset)
        click.echo(f"✓ Loaded {num_examples} examples")

    except Exception as e:
        click.secho(f"Error loading dataset: {e}", fg="red", err=True)
        sys.exit(1)

    # Set random seed
    random.seed(seed)

    def translate_preserving_links(text: str, translator: "TranslationService") -> str:
        """
        Translate text while preserving markdown link URLs.
        Link text is translated, but URLs are kept intact.
        """
        import re

        # Pattern to match markdown links: [text](url)
        pattern = r"(\[[^\]]*\]\()([^\)]+)(\))"

        # Find all links and their positions
        links = []
        for match in re.finditer(pattern, text):
            links.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "prefix": match.group(1),  # "[text]("
                    "url": match.group(2),  # the URL
                    "suffix": match.group(3),  # ")"
                    "full": match.group(0),
                }
            )

        if not links:
            # No links, translate the whole text
            return translator.translate_batch([text])[0]

        # Split text into segments: text before/between/after links
        segments = []
        last_end = 0

        for link in links:
            # Text before this link
            if link["start"] > last_end:
                segments.append({"type": "text", "content": text[last_end : link["start"]]})

            # The link itself - extract link text for translation
            link_text_match = re.match(r"\[([^\]]*)\]", link["prefix"])
            if link_text_match:
                link_text = link_text_match.group(1)
                segments.append(
                    {
                        "type": "link",
                        "link_text": link_text,
                        "url": link["url"],
                    }
                )

            last_end = link["end"]

        # Text after last link
        if last_end < len(text):
            segments.append({"type": "text", "content": text[last_end:]})

        # Translate all text segments and link texts together
        texts_to_translate = []
        for seg in segments:
            if seg["type"] == "text":
                texts_to_translate.append(seg["content"])
            elif seg["type"] == "link":
                texts_to_translate.append(seg["link_text"])

        # Batch translate
        if texts_to_translate:
            translated = translator.translate_batch(texts_to_translate)
        else:
            translated = []

        # Reassemble with translated text but original URLs
        result = []
        trans_idx = 0
        for seg in segments:
            if seg["type"] == "text":
                result.append(translated[trans_idx])
                trans_idx += 1
            elif seg["type"] == "link":
                # Use translated link text but original URL
                result.append(f"[{translated[trans_idx]}]({seg['url']})")
                trans_idx += 1

        return "".join(result)

    # Augmentation transform function
    def transform_batch(batch: dict, batch_indices: list[int]) -> dict:
        texts = []
        translated_count = 0

        for i in range(len(batch["corrected_agent_trace"])):
            row_list = batch["corrected_agent_trace"][i]
            full_prompt = row_list[0]["value"].replace("\r\n", "\n")
            parts = parse_corrected_agent_trace(full_prompt)

            # Deterministic decision based on seed + index
            random.seed(seed + batch_indices[i])
            should_translate = random.random() < ratio
            lang = random.choice(languages) if should_translate else None

            query = parts.get("user_query", "").strip()

            if should_translate and lang and query:
                translator = TranslationService(target_lang=lang)

                # Translate user query (preserving any markdown links)
                parts["user_query"] = translate_preserving_links(query, translator)

                # Translate final answer if present (preserving markdown links)
                comp = parts["assistant_completion"]
                marker = FINAL_ANSWER_PROMPT
                if marker not in comp and marker.endswith("\n"):
                    marker = marker.rstrip("\n")

                if marker in comp:
                    pre, final_ans = comp.rsplit(marker, 1)
                    if final_ans.strip():
                        trans_ans = translate_preserving_links(final_ans.strip(), translator)
                        parts["assistant_completion"] = f"{pre}{marker}{trans_ans}"

                translated_count += 1

            texts.append(build_full_prompt(parts))

        return {"text": texts}

    # Run augmentation
    click.echo("\n🔄 Augmenting dataset...")
    try:
        augmented_dataset = original_dataset.map(
            transform_batch,
            batched=True,
            batch_size=32,
            with_indices=True,
            load_from_cache_file=False,
            num_proc=1,
            desc="Augmenting",
        )

        # Count translations (approximate based on ratio)
        expected_translations = int(num_examples * ratio)
        click.echo(f"✓ Augmentation complete (~{expected_translations} samples translated)")

    except Exception as e:
        click.secho(f"Error during augmentation: {e}", fg="red", err=True)
        sys.exit(1)

    # Save augmented dataset
    click.echo(f"\n💾 Saving to {output}...")
    try:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        augmented_dataset.save_to_disk(str(output_path))
        click.secho(f"✅ Augmented dataset saved to {output}", fg="green")
    except Exception as e:
        click.secho(f"Error saving dataset: {e}", fg="red", err=True)
        sys.exit(1)

    # Cleanup GPU memory
    if torch.cuda.is_available():
        click.echo("\n🧹 Cleaning up GPU memory...")
        TranslationService._model_cache.clear()
        TranslationService._tokenizer_cache.clear()
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    click.echo("\n✅ Done!")


@dataset.command("augment-pipeline")
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True),
    required=True,
    help="JSON configuration file for augmentation pipeline",
)
@click.option(
    "--input",
    "-i",
    "input_override",
    type=str,
    default=None,
    help="Override input path from config",
)
@click.option(
    "--output",
    "-o",
    "output_override",
    type=str,
    default=None,
    help="Override output path from config",
)
@click.option(
    "--track-wandb",
    is_flag=True,
    default=False,
    help="Log augmented dataset to W&B as artifact",
)
@click.option(
    "--run-name",
    type=str,
    default=None,
    help="Custom W&B run name",
)
@click.pass_context
def augment_pipeline(
    ctx: click.Context,
    config_path: str,
    input_override: str | None,
    output_override: str | None,
    track_wandb: bool,
    run_name: str | None,
) -> None:
    """
    Augment dataset using a pipeline configuration file.

    This command implements FR-DATA-11 through FR-DATA-17, providing:
    - Dataset size multiplication (e.g., 3x)
    - Pipeline-based transformations with independent probabilities
    - Feature combination (multiple augmentations per variant)
    - Post-generation deduplication
    - Comprehensive metadata for reproducibility
    - Optional W&B tracking

    Examples:

        # Basic usage with config file
        irca dataset augment-pipeline -c configs/augment_full.json

        # Override input/output paths
        irca dataset augment-pipeline -c configs/augment.json -i datasets/v2 -o datasets/v2-aug

        # With W&B tracking
        irca dataset augment-pipeline -c configs/augment.json --track-wandb
    """
    import time
    from pathlib import Path

    from datasets import load_from_disk
    from src.augmentation.config import load_config
    from src.augmentation.metadata import generate_metadata, save_metadata
    from src.augmentation.pipeline import AugmentationPipeline

    settings = get_settings()
    verbose = ctx.obj.get("verbose", False)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load and validate config
    click.echo(f"📋 Loading config: {config_path}")
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        click.secho(f"Config error: {e}", fg="red", err=True)
        sys.exit(1)

    # Apply CLI overrides
    input_path = input_override or config.input
    output_path = output_override or config.output

    if not input_path:
        click.secho("Error: Input path required (in config or via --input)", fg="red", err=True)
        sys.exit(1)
    if not output_path:
        click.secho("Error: Output path required (in config or via --output)", fg="red", err=True)
        sys.exit(1)

    click.echo("\n🔧 Pipeline Configuration:")
    click.echo(f"   Input: {input_path}")
    click.echo(f"   Output: {output_path}")
    click.echo(f"   Multiply: {config.multiply}x")
    click.echo(f"   Seed: {config.seed}")
    click.echo(f"   Deduplicate: {config.deduplicate}")
    click.echo(f"   Pipeline steps: {len(config.pipeline)}")
    for i, step in enumerate(config.pipeline, 1):
        params_str = f" ({step.params})" if step.params else ""
        click.echo(f"      {i}. {step.type} (p={step.probability}){params_str}")

    # Load dataset
    click.echo("\n📊 Loading dataset...")
    try:
        input_path_resolved = Path(input_path)
        if not input_path_resolved.exists():
            # Try from settings datasets_path
            input_path_resolved = settings.datasets_path / input_path

        if not input_path_resolved.exists():
            click.secho(f"Error: Dataset not found at {input_path}", fg="red", err=True)
            sys.exit(1)

        dataset = load_from_disk(str(input_path_resolved))
        click.echo(f"   Loaded {len(dataset)} samples")

    except Exception as e:
        click.secho(f"Error loading dataset: {e}", fg="red", err=True)
        sys.exit(1)

    # Run augmentation pipeline
    click.echo("\n🔄 Running augmentation pipeline...")
    start_time = time.time()

    try:
        pipeline = AugmentationPipeline(config)
        augmented_dataset, statistics = pipeline.augment(dataset)
        duration = time.time() - start_time

        click.echo("\n📈 Statistics:")
        click.echo(f"   Input samples: {statistics['input_samples']}")
        click.echo(f"   Variants generated: {statistics['variants_generated']}")
        click.echo(f"   Duplicates removed: {statistics['duplicates_removed']}")
        click.echo(f"   Final output: {len(augmented_dataset)} samples")
        click.echo(f"   Duration: {duration:.1f}s")

        if statistics["augmentation_counts"]:
            click.echo("   Augmentation counts:")
            for aug_name, count in statistics["augmentation_counts"].items():
                click.echo(f"      {aug_name}: {count}")

    except Exception as e:
        click.secho(f"Error during augmentation: {e}", fg="red", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    # Save augmented dataset
    click.echo("\n💾 Saving augmented dataset...")
    output_path_resolved = Path(output_path)
    output_path_resolved.mkdir(parents=True, exist_ok=True)

    try:
        augmented_dataset.save_to_disk(str(output_path_resolved))
        click.echo(f"   Saved to: {output_path_resolved}")

    except Exception as e:
        click.secho(f"Error saving dataset: {e}", fg="red", err=True)
        sys.exit(1)

    # Generate and save metadata
    click.echo("\n📝 Generating metadata...")
    try:
        metadata = generate_metadata(
            config=config,
            input_path=str(input_path_resolved),
            output_path=str(output_path_resolved),
            statistics=statistics,
            duration_seconds=duration,
        )
        metadata_path = save_metadata(metadata, output_path_resolved)
        click.echo(f"   Saved to: {metadata_path}")

    except Exception as e:
        click.secho(f"Warning: Could not save metadata: {e}", fg="yellow")

    # W&B tracking
    if track_wandb:
        click.echo("\n📊 Logging to W&B...")
        try:
            import wandb

            # Initialize W&B if needed
            if settings.wandb_project:
                wandb.init(
                    project=settings.wandb_project,
                    entity=settings.wandb_entity,
                    name=run_name or f"augment-{Path(output_path).name}",
                    job_type="augmentation",
                    config=config.model_dump(),
                )

                # Log as artifact
                artifact = wandb.Artifact(
                    name=Path(output_path).name,
                    type="dataset",
                    metadata={
                        "config": config.model_dump(),
                        "statistics": statistics,
                        "input_path": str(input_path),
                    },
                )
                artifact.add_dir(str(output_path_resolved))
                wandb.log_artifact(artifact)

                wandb.finish()
                click.echo("   Logged as W&B artifact")

        except ImportError:
            click.secho("Warning: wandb not installed", fg="yellow")
        except Exception as e:
            click.secho(f"Warning: W&B logging failed: {e}", fg="yellow")

    click.echo("\n✅ Augmentation complete!")


@dataset.command("diversity")
@click.option(
    "--dataset",
    "-d",
    "dataset_path",
    type=str,
    default=None,
    help="Dataset to analyze (local path or HuggingFace ID)",
)
@click.option(
    "--original",
    "-o",
    "original_path",
    type=str,
    default=None,
    help="Original dataset for comparison",
)
@click.option(
    "--augmented",
    "-a",
    "augmented_path",
    type=str,
    default=None,
    help="Augmented dataset for comparison",
)
@click.option(
    "--model",
    "-m",
    "model_name",
    type=str,
    default=None,
    help="Model for perplexity computation (e.g., Qwen/Qwen2.5-3B)",
)
@click.option(
    "--quick",
    is_flag=True,
    default=False,
    help="Quick mode: only compute model-free metrics (Vendi + N-gram)",
)
@click.option(
    "--sample-size",
    type=int,
    default=None,
    help="Sample size for large datasets",
)
@click.option(
    "--include-semantic",
    is_flag=True,
    default=False,
    help="Include Vendi Score (requires sentence-transformers)",
)
@click.option(
    "--output",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable perplexity caching",
)
@click.option(
    "--report",
    "-r",
    "report_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Generate HTML report at this path (e.g., report.html)",
)
@click.option(
    "--eval-mode",
    type=click.Choice(["full", "completion"]),
    default="full",
    help="Perplexity evaluation mode: 'full' for entire sample, 'completion' for response only (recommended for SFT)",
)
@click.option(
    "--no-chat-template",
    "no_chat_template",
    is_flag=True,
    default=False,
    help="Disable chat template conversion. By default, IRCA-formatted text is converted to the model's native chat template for accurate perplexity.",
)
@click.option(
    "--baseline-model",
    "-b",
    "baseline_model",
    type=str,
    default=None,
    help="Baseline model for comparison (e.g., meta-llama/Llama-3.2-1B)",
)
@click.option(
    "--train-set",
    "train_path",
    type=str,
    default=None,
    help="Training set path for generalization evaluation",
)
@click.option(
    "--test-set",
    "test_path",
    type=str,
    default=None,
    help="Test set path for generalization evaluation",
)
@click.pass_context
def diversity(
    ctx: click.Context,
    dataset_path: str | None,
    original_path: str | None,
    augmented_path: str | None,
    model_name: str | None,
    quick: bool,
    sample_size: int | None,
    include_semantic: bool,
    output: str,
    no_cache: bool,
    report_path: str | None,
    eval_mode: str,
    no_chat_template: bool,
    baseline_model: str | None,
    train_path: str | None,
    test_path: str | None,
) -> None:
    """
    Analyze dataset diversity for SFT evaluation.

    Computes diversity metrics to evaluate augmentation effectiveness.
    Primary metric is perplexity-based (requires --model), with lexical
    diversity (Distinct-n) always computed.

    Modes:

    1. Single dataset analysis:

        irca dataset diversity -d datasets/my-data --model Qwen/Qwen2.5-3B

    2. Quick mode (no model required):

        irca dataset diversity -d datasets/my-data --quick

    3. Comparison mode (original vs augmented):

        irca dataset diversity -o datasets/original -a datasets/augmented --model Qwen/Qwen2.5-3B

    4. Model robustness evaluation (finetuned vs baseline):

        irca dataset diversity -d datasets/test --model /path/to/finetuned \\
            --baseline-model meta-llama/Llama-3.2-1B --eval-mode completion

    5. Generalization evaluation (train vs test):

        irca dataset diversity --train-set datasets/train --test-set datasets/test \\
            --model /path/to/finetuned --baseline-model base-model --eval-mode completion

    Examples:

        # Analyze single dataset with perplexity
        irca dataset diversity -d ./datasets/irca-v1 -m Qwen/Qwen2.5-3B

        # Quick analysis without model
        irca dataset diversity -d ./datasets/irca-v1 --quick --include-semantic

        # Compare original vs augmented
        irca dataset diversity -o ./datasets/original -a ./datasets/augmented -m gpt2

        # JSON output for scripting
        irca dataset diversity -d ./datasets/irca-v1 --quick --output json

        # Generate HTML report
        irca dataset diversity -o ./datasets/original -a ./datasets/augmented --quick --report report.html

        # Evaluate finetuned model robustness (completion-only mode)
        irca dataset diversity -d ./test_set --model ./finetuned \\
            --baseline-model meta-llama/Llama-3.2-1B --eval-mode completion

        # Evaluate generalization (train vs test)
        irca dataset diversity --train-set ./train --test-set ./test \\
            --model ./finetuned --baseline-model base --eval-mode completion
    """
    import json as json_module
    from pathlib import Path

    from datasets import load_from_disk
    from src.diversity.evaluation import evaluate_generalization, evaluate_model_robustness
    from src.diversity.lexical import compute_lexical_diversity
    from src.diversity.perplexity import compare_datasets, compute_perplexity_profile
    from src.diversity.utils import format_delta

    settings = get_settings()
    verbose = ctx.obj.get("verbose", False)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Validate arguments - determine mode
    comparison_mode = original_path is not None and augmented_path is not None
    single_mode = dataset_path is not None
    generalization_mode = train_path is not None and test_path is not None
    robustness_mode = single_mode and baseline_model is not None

    # Mode validation
    modes_active = sum(
        [
            comparison_mode,
            single_mode and not robustness_mode,  # Plain single mode (no baseline)
            robustness_mode,  # Single mode with baseline comparison
            generalization_mode,
        ]
    )
    if modes_active == 0:
        click.secho(
            "Error: Provide one of:\n"
            "  --dataset for single/robustness analysis\n"
            "  --original and --augmented for comparison\n"
            "  --train-set and --test-set for generalization evaluation",
            fg="red",
            err=True,
        )
        sys.exit(1)

    if comparison_mode and (single_mode or generalization_mode):
        click.secho(
            "Error: Cannot mix --original/--augmented with --dataset or --train-set/--test-set",
            fg="red",
            err=True,
        )
        sys.exit(1)

    if generalization_mode and single_mode:
        click.secho(
            "Error: Cannot use --dataset with --train-set/--test-set",
            fg="red",
            err=True,
        )
        sys.exit(1)

    if not quick and model_name is None:
        click.secho(
            "Error: --model is required unless --quick mode is used",
            fg="red",
            err=True,
        )
        sys.exit(1)

    if generalization_mode and model_name is None:
        click.secho(
            "Error: --model is required for generalization evaluation",
            fg="red",
            err=True,
        )
        sys.exit(1)

    # Helper to load dataset
    def load_dataset_texts(path: str, text_column: str = "text") -> list[str]:
        resolved_path = Path(path)
        if not resolved_path.exists():
            resolved_path = settings.datasets_path / path

        if not resolved_path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        ds = load_from_disk(str(resolved_path))
        if text_column not in ds.column_names:
            raise ValueError(f"Dataset missing '{text_column}' column. Has: {ds.column_names}")

        return ds[text_column]

    # Output helper
    results: dict = {}
    json_mode = output == "json"

    # Helper for conditional output (suppress when JSON mode)
    def log(msg: str, **kwargs) -> None:
        if not json_mode:
            click.echo(msg, **kwargs)

    try:
        if generalization_mode:
            # Generalization evaluation mode (train vs test)
            log("📊 Model Generalization Evaluation")
            log("=" * 60)
            log(f"\nEvaluation mode: {eval_mode}")
            log(f"Model: {model_name}")
            if baseline_model:
                log(f"Baseline: {baseline_model}")

            log("\nLoading datasets...")
            train_texts = load_dataset_texts(train_path)
            test_texts = load_dataset_texts(test_path)

            log(f"  Train set: {len(train_texts)} samples")
            log(f"  Test set: {len(test_texts)} samples")

            # Run generalization evaluation
            results = evaluate_generalization(
                train_texts=train_texts,
                test_texts=test_texts,
                finetuned_model=model_name,
                baseline_model=baseline_model,
                eval_mode=eval_mode,
                sample_size=sample_size,
                show_progress=not json_mode,
            )

            # Add metadata
            results["metadata"] = {
                "train_path": train_path,
                "test_path": test_path,
                "model": model_name,
                "baseline_model": baseline_model,
                "eval_mode": eval_mode,
            }

            # Output
            if json_mode:
                click.echo(json_module.dumps(results, indent=2, default=str))
            else:
                _print_generalization_results(results)

        elif robustness_mode:
            # Model robustness evaluation (single dataset, finetuned vs baseline)
            log("📊 Model Robustness Evaluation")
            log("=" * 60)
            log(f"\nEvaluation mode: {eval_mode}")
            log(f"Finetuned model: {model_name}")
            log(f"Baseline model: {baseline_model}")

            log(f"\nLoading dataset: {dataset_path}")
            texts = load_dataset_texts(dataset_path)
            log(f"  Samples: {len(texts)}")

            # Run robustness evaluation
            results = evaluate_model_robustness(
                texts=texts,
                finetuned_model=model_name,
                baseline_model=baseline_model,
                eval_mode=eval_mode,
                sample_size=sample_size,
                show_progress=not json_mode,
            )

            # Add metadata
            results["metadata"] = {
                "dataset_path": dataset_path,
                "model": model_name,
                "baseline_model": baseline_model,
                "eval_mode": eval_mode,
            }

            # Output
            if json_mode:
                click.echo(json_module.dumps(results, indent=2, default=str))
            else:
                _print_robustness_results(results)

        elif comparison_mode:
            # Comparison mode (original vs augmented)
            log("📊 Dataset Diversity Comparison")
            log("=" * 60)

            log("\nLoading datasets...")
            original_texts = load_dataset_texts(original_path)
            augmented_texts = load_dataset_texts(augmented_path)

            log(f"  Original: {len(original_texts)} samples")
            log(f"  Augmented: {len(augmented_texts)} samples")

            # Run comparison
            apply_chat_template = not no_chat_template
            results = compare_datasets(
                original_texts=original_texts,
                augmented_texts=augmented_texts,
                model_name=model_name,
                quick=quick,
                sample_size=sample_size,
                show_progress=not json_mode,
                eval_mode=eval_mode,
                apply_chat_template=apply_chat_template,
            )

            # Add metadata for report generation
            results["metadata"] = {
                "original_path": original_path,
                "augmented_path": augmented_path,
                "model": model_name,
                "quick_mode": quick,
                "eval_mode": eval_mode,
                "apply_chat_template": apply_chat_template,
            }

            # Include semantic if requested
            if include_semantic:
                try:
                    from src.diversity.semantic import compute_vendi_score

                    log("\nComputing Vendi Score...")
                    results["original"]["vendi_score"] = compute_vendi_score(
                        original_texts, sample_size=sample_size or 5000, show_progress=not json_mode
                    )
                    results["augmented"]["vendi_score"] = compute_vendi_score(
                        augmented_texts, sample_size=sample_size or 5000, show_progress=not json_mode
                    )
                    results["delta"]["vendi_score_change"] = format_delta(
                        results["original"]["vendi_score"],
                        results["augmented"]["vendi_score"],
                    )
                except ImportError:
                    if not json_mode:
                        click.secho(
                            "Warning: sentence-transformers not installed, skipping Vendi Score",
                            fg="yellow",
                        )

            # Output
            if output == "json":
                click.echo(json_module.dumps(results, indent=2, default=str))
            else:
                _print_comparison_results(results, original_path, augmented_path, model_name, quick)

        else:
            # Single dataset mode
            log("📊 Dataset Diversity Analysis")
            log("=" * 60)

            log(f"\nLoading dataset: {dataset_path}")
            texts = load_dataset_texts(dataset_path)
            log(f"  Samples: {len(texts)}")

            results = {
                "dataset": dataset_path,
                "sample_count": len(texts),
            }

            # Lexical diversity (always)
            log("\nComputing lexical diversity...")
            results["lexical"] = compute_lexical_diversity(texts)

            # Perplexity (if model provided)
            if not quick and model_name:
                apply_chat_template = not no_chat_template
                mode_desc = "completion" if eval_mode == "completion" else "full sample"
                template_desc = " (with chat template)" if apply_chat_template else ""
                log(f"\nComputing {mode_desc} perplexity{template_desc} with {model_name}...")
                results["perplexity"] = compute_perplexity_profile(
                    texts,
                    model_name=model_name,
                    sample_size=sample_size,
                    use_cache=not no_cache,
                    show_progress=not json_mode,
                    eval_mode=eval_mode,
                    apply_chat_template=apply_chat_template,
                )

            # Semantic (if requested)
            if include_semantic:
                try:
                    from src.diversity.semantic import compute_semantic_diversity

                    log("\nComputing semantic diversity...")
                    results["semantic"] = compute_semantic_diversity(
                        texts, sample_size=sample_size or 5000, show_progress=not json_mode
                    )
                except ImportError:
                    if not json_mode:
                        click.secho(
                            "Warning: sentence-transformers not installed, skipping semantic metrics",
                            fg="yellow",
                        )

            # Output
            if json_mode:
                click.echo(json_module.dumps(results, indent=2, default=str))
            else:
                _print_single_results(results, dataset_path, model_name, quick)

        # Generate HTML report if requested
        if report_path:
            from src.diversity.report import generate_report

            log(f"\nGenerating HTML report: {report_path}")
            report_file = generate_report(
                metrics=results,
                output_path=Path(report_path),
                title="IRCA Diversity Report",
            )
            # Write to stderr in JSON mode to avoid breaking JSON output
            click.secho(f"Report generated: {report_file}", fg="green", err=json_mode)

    except FileNotFoundError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def _print_single_results(results: dict, dataset_path: str, model_name: str | None, quick: bool) -> None:
    """Print single dataset analysis results."""
    click.echo(f"\n{'=' * 60}")
    click.echo(f"Dataset: {dataset_path}")
    click.echo(f"Samples: {results['sample_count']}")
    click.echo(f"{'=' * 60}")

    # Lexical
    click.echo("\n📝 Lexical Diversity:")
    lex = results["lexical"]
    click.echo(f"  Distinct-1: {lex['distinct_1']:.4f}")
    click.echo(f"  Distinct-2: {lex['distinct_2']:.4f}")
    click.echo(f"  Distinct-3: {lex['distinct_3']:.4f}")
    if "type_token_ratio" in lex:
        click.echo(f"  Type-Token Ratio: {lex['type_token_ratio']:.4f}")

    # Perplexity
    if "perplexity" in results:
        click.echo(f"\n📈 Perplexity Profile (model: {model_name}):")
        ppl = results["perplexity"]
        click.echo(f"  Mean: {ppl['mean_perplexity']:.2f}")
        click.echo(f"  Std: {ppl['std_perplexity']:.2f}")
        click.echo(f"  Min: {ppl['min_perplexity']:.2f}")
        click.echo(f"  Max: {ppl['max_perplexity']:.2f}")
        click.echo(f"  Median: {ppl['median_perplexity']:.2f}")
        click.echo(f"  P90: {ppl['p90_perplexity']:.2f}")

    # Semantic
    if "semantic" in results:
        click.echo("\n🧠 Semantic Diversity:")
        sem = results["semantic"]
        click.echo(f"  Vendi Score: {sem['vendi_score']:.2f}")
        click.echo(f"  Effective Diversity: {sem['effective_diversity_ratio']:.1%}")
        click.echo(f"  Avg Pairwise Distance: {sem['avg_pairwise_distance']:.4f}")

    click.echo("")


def _print_comparison_results(
    results: dict,
    original_path: str,
    augmented_path: str,
    model_name: str | None,
    quick: bool,
) -> None:
    """Print comparison results."""
    click.echo(f"\n{'=' * 60}")
    click.echo(f"Original: {original_path} ({results['original']['sample_count']} samples)")
    click.echo(f"Augmented: {augmented_path} ({results['augmented']['sample_count']} samples)")
    click.echo(f"Size change: {results['delta']['sample_count_change']}")
    click.echo(f"{'=' * 60}")

    # Lexical comparison
    click.echo("\n📝 Lexical Diversity:")
    click.echo(f"  {'Metric':<20} {'Original':>12} {'Augmented':>12} {'Change':>12}")
    click.echo(f"  {'-' * 56}")

    orig_lex = results["original"]["lexical"]
    aug_lex = results["augmented"]["lexical"]

    for key in ["distinct_1", "distinct_2", "distinct_3"]:
        orig_val = orig_lex[key]
        aug_val = aug_lex[key]
        change = results["delta"].get(f"{key}_change", "N/A")
        click.echo(f"  {key:<20} {orig_val:>12.4f} {aug_val:>12.4f} {change:>12}")

    # Perplexity comparison
    if "perplexity" in results["original"]:
        click.echo(f"\n📈 Perplexity Profile (model: {model_name}):")
        click.echo(f"  {'Metric':<20} {'Original':>12} {'Augmented':>12} {'Change':>12}")
        click.echo(f"  {'-' * 56}")

        orig_ppl = results["original"]["perplexity"]
        aug_ppl = results["augmented"]["perplexity"]

        for key, label in [
            ("mean_perplexity", "Mean"),
            ("std_perplexity", "Std"),
            ("p90_perplexity", "P90"),
        ]:
            orig_val = orig_ppl[key]
            aug_val = aug_ppl[key]
            change = results["delta"].get(f"{key}_change", "N/A")
            if change == "N/A":
                from src.diversity.utils import format_delta

                change = format_delta(orig_val, aug_val)
            click.echo(f"  {label:<20} {orig_val:>12.2f} {aug_val:>12.2f} {change:>12}")

    # Semantic comparison
    if "vendi_score" in results.get("original", {}):
        click.echo("\n🧠 Semantic Diversity:")
        orig_vendi = results["original"]["vendi_score"]
        aug_vendi = results["augmented"]["vendi_score"]
        change = results["delta"].get("vendi_score_change", "N/A")
        click.echo(f"  {'Vendi Score':<20} {orig_vendi:>12.2f} {aug_vendi:>12.2f} {change:>12}")

    # Summary
    if "summary" in results:
        click.echo("\n💡 Summary:")
        for line in results["summary"].split("\n"):
            click.echo(f"  {line}")

    click.echo("")


def _print_robustness_results(results: dict) -> None:
    """Print model robustness evaluation results."""
    click.echo(f"\n{'=' * 60}")
    click.echo("Model Robustness Evaluation")
    click.echo(f"{'=' * 60}")

    eval_mode = results.get("eval_mode", "full")
    mode_label = "Completion-only" if eval_mode == "completion" else "Full sample"
    click.echo(f"\nEvaluation mode: {mode_label}")
    click.echo(f"Samples evaluated: {results.get('sample_count', 'N/A')}")

    # Finetuned model results
    click.echo("\n📈 Finetuned Model Perplexity:")
    ft = results["finetuned"]
    click.echo(f"  Mean: {ft['mean_perplexity']:.2f}")
    click.echo(f"  Std: {ft['std_perplexity']:.2f}")
    click.echo(f"  P90: {ft['p90_perplexity']:.2f}")

    # Baseline comparison
    if "baseline" in results:
        click.echo("\n📉 Baseline Model Perplexity:")
        base = results["baseline"]
        click.echo(f"  Mean: {base['mean_perplexity']:.2f}")
        click.echo(f"  Std: {base['std_perplexity']:.2f}")
        click.echo(f"  P90: {base['p90_perplexity']:.2f}")

        click.echo("\n🎯 Improvement:")
        imp = results["improvement"]
        click.echo(f"  Perplexity change: {imp['mean_perplexity_change']}")

    # Summary
    if "summary" in results:
        click.echo("\n💡 Summary:")
        for line in results["summary"].split("\n"):
            click.echo(f"  {line}")

    click.echo("")


def _print_generalization_results(results: dict) -> None:
    """Print generalization evaluation results."""
    click.echo(f"\n{'=' * 60}")
    click.echo("Model Generalization Evaluation")
    click.echo(f"{'=' * 60}")

    eval_mode = results.get("eval_mode", "full")
    mode_label = "Completion-only" if eval_mode == "completion" else "Full sample"
    click.echo(f"\nEvaluation mode: {mode_label}")
    click.echo(f"Train samples: {results.get('train_count', 'N/A')}")
    click.echo(f"Test samples: {results.get('test_count', 'N/A')}")

    # Finetuned model results
    click.echo("\n📈 Finetuned Model:")
    ft = results["finetuned"]
    click.echo(f"  {'Dataset':<12} {'Mean PPL':>12} {'Std':>12}")
    click.echo(f"  {'-' * 36}")
    click.echo(f"  {'Train':<12} {ft['train']['mean_perplexity']:>12.2f} {ft['train']['std_perplexity']:>12.2f}")
    click.echo(f"  {'Test':<12} {ft['test']['mean_perplexity']:>12.2f} {ft['test']['std_perplexity']:>12.2f}")
    click.echo(f"  {'Gap':<12} {ft['gap']['relative']:>12}")

    # Baseline comparison
    if "baseline" in results:
        click.echo("\n📉 Baseline Model:")
        base = results["baseline"]
        click.echo(f"  {'Dataset':<12} {'Mean PPL':>12} {'Std':>12}")
        click.echo(f"  {'-' * 36}")
        click.echo(
            f"  {'Train':<12} {base['train']['mean_perplexity']:>12.2f} {base['train']['std_perplexity']:>12.2f}"
        )
        click.echo(f"  {'Test':<12} {base['test']['mean_perplexity']:>12.2f} {base['test']['std_perplexity']:>12.2f}")
        click.echo(f"  {'Gap':<12} {base['gap']['relative']:>12}")

        click.echo("\n🎯 Comparison:")
        comp = results["comparison"]
        click.echo(f"  Train improvement: {comp['train_improvement']}")
        click.echo(f"  Test improvement: {comp['test_improvement']}")
        click.echo(f"  Gap ratio: {comp['gap_ratio']:.2f}x baseline")

    # Summary
    if "summary" in results:
        click.echo("\n💡 Summary:")
        for line in results["summary"].split("\n"):
            click.echo(f"  {line}")

    click.echo("")


@dataset.command("combine")
@click.option(
    "--sources",
    "-s",
    type=str,
    multiple=True,
    required=True,
    help="Source dataset names (can specify multiple)",
)
@click.option(
    "--target",
    "-t",
    type=str,
    required=True,
    help="Target combined dataset name",
)
@click.option(
    "--workspace",
    "-w",
    type=str,
    default="irca_agent",
    help="Argilla workspace",
)
@click.pass_context
def combine(
    ctx: click.Context,
    sources: tuple[str, ...],
    target: str,
    workspace: str,
) -> None:
    """
    Combine multiple datasets into one.

    Merges multiple Argilla datasets into a single accumulated dataset.

    Examples:

        irca dataset combine -s dataset_v1 -s dataset_v2 -t combined_dataset
    """
    settings = get_settings()

    try:
        import argilla as rg

        click.echo("Connecting to Argilla...")
        rg.init(
            api_url=settings.argilla_api_url,
            api_key=settings.argilla_api_key,
        )

        all_records = []

        for source in sources:
            click.echo(f"Loading {source}...")
            try:
                ds = rg.FeedbackDataset.from_argilla(name=source, workspace=workspace)
                records = [r for r in ds.records if r.status == "submitted"]
                all_records.extend(records)
                click.echo(f"  Added {len(records)} records")
            except Exception as e:
                click.secho(f"  Warning: Could not load {source}: {e}", fg="yellow")

        click.echo(f"\nTotal records: {len(all_records)}")

        if not all_records:
            click.secho("No records to combine!", fg="yellow")
            return

        # Create target dataset (would need schema from source)
        click.echo(f"Creating combined dataset: {target}")
        click.secho("Note: Full implementation requires schema handling", fg="yellow")

    except ImportError:
        click.secho("Error: Argilla not installed", fg="red", err=True)
        sys.exit(1)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


@dataset.command("format")
@click.option(
    "--input",
    "-i",
    "input_path",
    type=str,
    required=True,
    help="Input dataset (local path or HuggingFace ID)",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    required=True,
    help="Output path for formatted dataset",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Augmentation config file (JSON)",
)
@click.option(
    "--source-column",
    type=str,
    default="corrected_agent_trace",
    help="Column containing raw prompt data",
)
@click.option(
    "--no-augment",
    is_flag=True,
    default=False,
    help="Disable all augmentations (even from config)",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility",
)
@click.pass_context
def format_dataset_cmd(
    ctx: click.Context,
    input_path: str,
    output_path: str,
    config_path: str | None,
    source_column: str,
    no_augment: bool,
    seed: int,
) -> None:
    """
    Format raw structured dataset to training-ready text.

    Converts datasets with structured columns (like 'corrected_agent_trace')
    to datasets with a 'text' column ready for finetuning or evaluation.

    This is the primary entry point for preparing data for:
    - Finetuning (produces 'text' column)
    - Diversity evaluation (produces comparable baseline/augmented datasets)

    Modes:

    1. Baseline (no augmentation):

        irca dataset format -i raw-dataset -o formatted-baseline --no-augment

    2. With augmentation config:

        irca dataset format -i raw-dataset -o formatted-augmented -c augment.json

    3. Custom source column:

        irca dataset format -i dataset -o formatted --source-column agent_trace

    Examples:

        # Format baseline for diversity comparison
        irca dataset format -i datasets/irca-raw -o datasets/baseline --no-augment

        # Format with augmentation for training
        irca dataset format -i datasets/irca-raw -o datasets/train -c configs/augment.json

        # Compare diversity after formatting
        irca dataset diversity -o datasets/baseline -a datasets/train --quick

    Config file format (augment.json):

        {
          "seed": 42,
          "multiply": 3,
          "structural": [
            {"type": "translate", "probability": 0.5, "params": {"languages": ["fr", "es"]}}
          ],
          "formatting": [
            {"type": "shuffle_functions", "probability": 0.8}
          ],
          "presentation": [
            {"type": "format_variation", "probability": 0.3}
          ]
        }
    """
    import time
    from pathlib import Path

    from datasets import load_from_disk
    from src.formatting import DatasetFormatter, load_format_config
    from src.formatting.config import create_baseline_config

    settings = get_settings()
    verbose = ctx.obj.get("verbose", False)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    click.echo("📋 Dataset Formatting")
    click.echo("=" * 60)

    # Load or create config
    if config_path and not no_augment:
        click.echo(f"\n📄 Loading config: {config_path}")
        try:
            config = load_format_config(config_path)
        except (FileNotFoundError, ValueError) as e:
            click.secho(f"Config error: {e}", fg="red", err=True)
            sys.exit(1)
    else:
        config = create_baseline_config(source_column)
        config.seed = seed

    # Apply CLI overrides
    config.source_column = source_column
    if no_augment:
        config.structural = []
        config.formatting = []
        config.presentation = []
        click.echo("\n⚠️  Augmentations disabled (--no-augment)")

    # Display config
    click.echo("\n🔧 Configuration:")
    click.echo(f"   Input: {input_path}")
    click.echo(f"   Output: {output_path}")
    click.echo(f"   Source column: {config.source_column}")
    click.echo(f"   Seed: {config.seed}")
    click.echo(f"   Multiply: {config.multiply}x")

    if config.has_augmentations():
        click.echo("   Augmentation steps:")
        for stage, step in config.get_all_steps():
            params_str = f" ({step.params})" if step.params else ""
            click.echo(f"      [{stage}] {step.type} (p={step.probability}){params_str}")
    else:
        click.echo("   Augmentations: None (baseline mode)")

    # Load dataset
    click.echo("\n📊 Loading dataset...")
    try:
        resolved_path = Path(input_path)
        if not resolved_path.exists():
            resolved_path = settings.datasets_path / input_path

        if not resolved_path.exists():
            click.secho(f"Error: Dataset not found: {input_path}", fg="red", err=True)
            sys.exit(1)

        dataset = load_from_disk(str(resolved_path))
        click.echo(f"   Loaded {len(dataset)} samples")
        click.echo(f"   Columns: {dataset.column_names}")

        # Verify source column exists
        if config.source_column not in dataset.column_names:
            click.secho(
                f"Error: Source column '{config.source_column}' not found. Available: {dataset.column_names}",
                fg="red",
                err=True,
            )
            sys.exit(1)

    except Exception as e:
        click.secho(f"Error loading dataset: {e}", fg="red", err=True)
        sys.exit(1)

    # Format dataset
    click.echo("\n🔄 Formatting dataset...")
    start_time = time.time()

    try:
        formatter = DatasetFormatter(config)
        formatted_dataset, statistics = formatter.format(dataset)
        duration = time.time() - start_time

        click.echo("\n📈 Statistics:")
        click.echo(f"   Input samples: {statistics['input_samples']}")
        click.echo(f"   Output samples: {statistics['output_samples']}")
        if statistics["variants_generated"] != statistics["output_samples"]:
            click.echo(f"   Variants generated: {statistics['variants_generated']}")
            click.echo(f"   Duplicates removed: {statistics['duplicates_removed']}")
        if statistics["parse_errors"] > 0:
            click.secho(f"   Parse errors: {statistics['parse_errors']}", fg="yellow")
        click.echo(f"   Duration: {duration:.1f}s")

        if statistics["augmentation_counts"]:
            click.echo("   Augmentation counts:")
            for aug_name, count in statistics["augmentation_counts"].items():
                click.echo(f"      {aug_name}: {count}")

    except Exception as e:
        click.secho(f"Error during formatting: {e}", fg="red", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    # Save formatted dataset
    click.echo("\n💾 Saving formatted dataset...")
    try:
        output_resolved = Path(output_path)
        output_resolved.mkdir(parents=True, exist_ok=True)
        formatted_dataset.save_to_disk(str(output_resolved))
        click.echo(f"   Saved to: {output_resolved}")

    except Exception as e:
        click.secho(f"Error saving dataset: {e}", fg="red", err=True)
        sys.exit(1)

    # Show sample
    if len(formatted_dataset) > 0:
        click.echo("\n📝 Sample output (first 300 chars):")
        sample_text = formatted_dataset[0]["text"][:300]
        click.echo(f"   {sample_text}...")

    click.echo("\n✅ Formatting complete!")
    click.echo(f"   Ready for: irca dataset diversity -d {output_path} --quick")


@dataset.command("split")
@click.option(
    "--input",
    "-i",
    "input_path",
    type=str,
    required=True,
    help="Input dataset (local path)",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(),
    required=True,
    help="Output directory for train/test splits",
)
@click.option(
    "--train-ratio",
    type=float,
    default=0.8,
    help="Ratio of data for training (default: 0.8)",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility",
)
@click.option(
    "--stratify-column",
    type=str,
    default=None,
    help="Column to stratify split by (optional)",
)
@click.pass_context
def split_dataset_cmd(
    ctx: click.Context,
    input_path: str,
    output_dir: str,
    train_ratio: float,
    seed: int,
    stratify_column: str | None,
) -> None:
    """
    Split dataset into train/test sets for controlled experiments.

    Creates reproducible train/test splits from an original dataset.
    This is essential for measuring augmentation impact - always split
    BEFORE augmentation to avoid data leakage.

    Output structure:

        output_dir/
        ├── train/              # Training split (raw, not augmented)
        ├── test/               # Test split (raw, not augmented)
        └── split_info.json     # Metadata about the split

    Workflow for augmentation experiments:

        1. Split original dataset:
           irca dataset split -i datasets/original -o experiments/exp001

        2. Format train WITHOUT augmentation:
           irca dataset format -i experiments/exp001/train \\
               -o experiments/exp001/train_formatted --no-augment

        3. Format train WITH augmentation:
           irca dataset format -i experiments/exp001/train \\
               -o experiments/exp001/train_augmented -c configs/augment.json

        4. Format test (same for both, or create augmented version):
           irca dataset format -i experiments/exp001/test \\
               -o experiments/exp001/test_formatted --no-augment

        5. Finetune models and compare on test set

    Examples:

        # Basic 80/20 split
        irca dataset split -i datasets/irca_v5 -o experiments/exp001

        # 90/10 split with specific seed
        irca dataset split -i datasets/irca_v5 -o experiments/exp002 \\
            --train-ratio 0.9 --seed 123

    Statistical note:

        With 190 samples and 80/20 split:
        - Train: ~152 samples
        - Test: ~38 samples

        Test set is small - consider using confidence intervals
        and effect sizes rather than just point estimates.
    """
    import json
    import os
    from datetime import datetime
    from pathlib import Path

    from datasets import load_from_disk

    verbose = ctx.obj.get("verbose", False)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    click.echo("✂️  Dataset Splitting")
    click.echo("=" * 60)

    # Validate train ratio
    if not 0.1 <= train_ratio <= 0.95:
        click.secho("Error: train-ratio must be between 0.1 and 0.95", fg="red", err=True)
        sys.exit(1)

    # Load input dataset
    click.echo(f"\n📂 Loading dataset: {input_path}")
    try:
        if os.path.exists(input_path):
            dataset = load_from_disk(input_path)
        else:
            click.secho(f"Error: Dataset not found at {input_path}", fg="red", err=True)
            sys.exit(1)

        # Handle DatasetDict
        if hasattr(dataset, "keys"):
            if "train" in dataset:
                dataset = dataset["train"]
                click.echo("   (Extracted 'train' split from DatasetDict)")
            else:
                click.secho(
                    f"Error: DatasetDict has no 'train' split. Available: {list(dataset.keys())}",
                    fg="red",
                    err=True,
                )
                sys.exit(1)

        total_samples = len(dataset)
        click.echo(f"   Total samples: {total_samples}")
        click.echo(f"   Columns: {dataset.column_names}")

    except Exception as e:
        click.secho(f"Error loading dataset: {e}", fg="red", err=True)
        sys.exit(1)

    # Calculate split sizes
    train_size = int(total_samples * train_ratio)
    test_size = total_samples - train_size

    click.echo("\n📊 Split configuration:")
    click.echo(f"   Train ratio: {train_ratio:.0%}")
    click.echo(f"   Train samples: {train_size}")
    click.echo(f"   Test samples: {test_size}")
    click.echo(f"   Seed: {seed}")

    if test_size < 20:
        click.secho(
            f"\n⚠️  Warning: Test set has only {test_size} samples. Statistical significance may be limited.",
            fg="yellow",
        )

    # Perform split
    click.echo("\n🔀 Splitting dataset...")
    try:
        # Shuffle and split
        shuffled = dataset.shuffle(seed=seed)

        train_dataset = shuffled.select(range(train_size))
        test_dataset = shuffled.select(range(train_size, total_samples))

        click.echo(f"   Train: {len(train_dataset)} samples")
        click.echo(f"   Test: {len(test_dataset)} samples")

    except Exception as e:
        click.secho(f"Error during split: {e}", fg="red", err=True)
        sys.exit(1)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train"
    test_path = output_path / "test"

    # Save datasets
    click.echo("\n💾 Saving splits...")
    try:
        train_dataset.save_to_disk(str(train_path))
        click.echo(f"   Train saved to: {train_path}")

        test_dataset.save_to_disk(str(test_path))
        click.echo(f"   Test saved to: {test_path}")

    except Exception as e:
        click.secho(f"Error saving splits: {e}", fg="red", err=True)
        sys.exit(1)

    # Save split metadata
    split_info = {
        "created_at": datetime.now().isoformat(),
        "input_path": str(input_path),
        "total_samples": total_samples,
        "train_ratio": train_ratio,
        "train_samples": train_size,
        "test_samples": test_size,
        "seed": seed,
        "stratify_column": stratify_column,
        "columns": dataset.column_names,
        "train_path": str(train_path),
        "test_path": str(test_path),
    }

    split_info_path = output_path / "split_info.json"
    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    click.echo(f"   Metadata saved to: {split_info_path}")

    # Summary
    click.echo("\n✅ Split complete!")
    click.echo("\n📋 Next steps for augmentation experiment:")
    click.echo("   1. Format train (no augmentation):")
    click.echo(f"      irca dataset format -i {train_path} -o {output_path}/train_formatted --no-augment")
    click.echo("   2. Format train (with augmentation):")
    click.echo(
        f"      irca dataset format -i {train_path} -o {output_path}/train_augmented -c configs/format_full_augment.json"
    )
    click.echo("   3. Format test (for evaluation):")
    click.echo(f"      irca dataset format -i {test_path} -o {output_path}/test_formatted --no-augment")
