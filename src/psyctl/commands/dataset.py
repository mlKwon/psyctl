"""Dataset generation commands."""

from pathlib import Path

import click
from rich.console import Console

from psyctl.core.dataset_builder import DatasetBuilder
from psyctl.core.logger import get_logger

console = Console()
logger = get_logger("dataset")


@click.command()
@click.option("--model", required=True, help="Model name (e.g., google/gemma-3-27b-it)")
@click.option(
    "--personality",
    required=True,
    help="Personality traits (e.g., Extroversion, Machiavellism)",
)
@click.option(
    "--output", required=True, type=click.Path(), help="Output directory path"
)
def build_caa(model: str, personality: str, output: str):
    """Build CAA dataset for steering vector extraction."""
    logger.info("Starting CAA dataset build")
    logger.info(f"Model: {model}")
    logger.info(f"Personality: {personality}")
    logger.info(f"Output: {output}")

    console.print(f"[blue]Building CAA dataset...[/blue]")
    console.print(f"Model: {model}")
    console.print(f"Personality: {personality}")
    console.print(f"Output: {output}")

    try:
        builder = DatasetBuilder()
        builder.build_caa_dataset(model, personality, Path(output))

        logger.success(f"Dataset built successfully at {output}")
        console.print(f"[green]Dataset built successfully at {output}[/green]")
    except Exception as e:
        logger.error(f"Failed to build dataset: {e}")
        raise
