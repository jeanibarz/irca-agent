"""
Generate Command

Commands for generating agent traces from user queries.
"""

import logging
import sys
from pathlib import Path

import click

from src.config import get_settings

logger = logging.getLogger(__name__)


@click.group()
def generate() -> None:
    """Generate agent traces for dataset creation."""
    pass


@generate.command("traces")
@click.option(
    "--model",
    "-m",
    type=str,
    default=None,
    help="Path to model or HuggingFace model ID",
)
@click.option(
    "--dataset",
    "-d",
    type=str,
    default="irca_user_query_dataset_v5-6",
    help="Name of the source dataset (in datasets directory)",
)
@click.option(
    "--start",
    "-s",
    type=int,
    default=0,
    help="Starting index in the dataset",
)
@click.option(
    "--limit",
    "-n",
    type=int,
    default=None,
    help="Maximum number of traces to generate",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file for generated traces",
)
@click.pass_context
def traces(
    ctx: click.Context,
    model: str | None,
    dataset: str,
    start: int,
    limit: int | None,
    output: str | None,
) -> None:
    """
    Generate agent traces from user queries.

    Reads user queries from a dataset and generates complete agent traces
    using constrained generation with the specified model.

    Examples:

        # Generate traces with default model
        irca generate traces

        # Generate 10 traces starting from index 5
        irca generate traces --start 5 --limit 10

        # Use a specific model
        irca generate traces --model mistralai/Mistral-7B-v0.1
    """
    from datasets import load_from_disk  # type: ignore
    from src.core.generation import TraceGenerator
    from src.core.utils import shuffle_json_functions

    settings = get_settings()
    verbose = ctx.obj.get("verbose", False)

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Determine model path
    if model:
        model_path = model
    else:
        model_path = str(
            settings.finetuned_models_path
            / "Mistral-7B-Instruct-v0.2-with-data-augmentation_irca_agent_v5-6.gguf"
            / "checkpoint-97"
        )
        click.echo(f"Using default model: {model_path}")

    # Load source dataset
    dataset_path = settings.datasets_path / dataset
    click.echo(f"Loading dataset from: {dataset_path}")

    try:
        src_ds = load_from_disk(str(dataset_path))
        click.echo(f"Loaded {len(src_ds)} records from dataset")
    except Exception as e:
        click.secho(f"Error: Failed to load dataset: {e}", fg="red", err=True)
        sys.exit(1)

    # Initialize trace generator
    click.echo("Initializing trace generator...")
    try:
        trace_generator = TraceGenerator(model_name_or_path=model_path)
    except Exception as e:
        click.secho(f"Error: Failed to initialize model: {e}", fg="red", err=True)
        sys.exit(1)

    # Process records
    records = src_ds
    start_idx = start
    end_idx = len(records) if limit is None else min(start + limit, len(records))

    click.echo(f"Generating traces for records {start_idx} to {end_idx - 1}")
    generated_traces = []

    with click.progressbar(
        range(start_idx, end_idx),
        label="Generating traces",
        show_pos=True,
    ) as progress:
        for i in progress:
            try:
                available_functions = records[i]["available_functions"]
                user_query = records[i]["corrected_user_query"][0]["value"]
            except (KeyError, IndexError) as e:
                logger.warning(f"Skipping record {i}: {e}")
                continue

            # Shuffle functions to reduce positional bias
            shuffled_functions = shuffle_json_functions(available_functions)

            # Generate trace
            trace = trace_generator.generate_single_trace(
                available_functions=shuffled_functions,
                user_query=user_query,
            )
            generated_traces.append(
                {
                    "index": i,
                    "user_query": user_query,
                    "trace": trace.to_string(),
                }
            )

    click.echo(f"\n✅ Generated {len(generated_traces)} traces")

    # Save output if specified
    if output:
        import json

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(generated_traces, f, indent=2)
        click.echo(f"Saved traces to: {output_path}")


@generate.command("user-queries")
@click.option(
    "--model",
    "-m",
    type=str,
    default=None,
    help="Path to model for query generation",
)
@click.option(
    "--count",
    "-n",
    type=int,
    default=10,
    help="Number of queries to generate",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file for generated queries",
)
def user_queries(
    model: str | None,
    count: int,
    output: str | None,
) -> None:
    """
    Generate synthetic user queries.

    Creates diverse user queries based on available functions
    for use in trace generation.
    """
    click.echo("User query generation not yet implemented")
    click.echo("Use scripts/main_generator_user_query.py for now")
