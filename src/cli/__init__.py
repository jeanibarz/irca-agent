"""
IRCA-Agent Command Line Interface

Main entry point for the IRCA-Agent toolkit.

Usage:
    irca generate --help
    irca finetune --help
    irca dataset --help
"""

import click

from .commands import dataset, finetune, generate


@click.group()
@click.version_option(version="0.1.0", prog_name="irca")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """
    IRCA-Agent: Dataset generation toolkit for function-calling agents.

    Generate high-quality datasets for finetuning language models
    with function-calling / tool-use capabilities.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# Register command groups
cli.add_command(generate.generate)
cli.add_command(finetune.finetune)
cli.add_command(dataset.dataset)


def main() -> None:
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
