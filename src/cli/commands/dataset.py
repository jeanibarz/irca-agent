"""
Dataset Command

Commands for managing datasets (Argilla, HuggingFace Hub).
"""

import logging
import sys

import click

from config import get_settings

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

    from datasets import load_dataset

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

        click.echo(f"Connecting to Argilla...")
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
