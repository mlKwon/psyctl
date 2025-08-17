"""Steering vector extraction commands."""

from pathlib import Path

import click
from rich.console import Console

from psyctl.core.logger import get_logger
from psyctl.core.steering_extractor import SteeringExtractor

console = Console()
logger = get_logger("extract")


@click.command()
@click.option("--model", required=True, help="Model name")
@click.option(
    "--layer", required=True, help="Target layer (e.g., model.layers[13].mlp.down_proj)"
)
@click.option(
    "--dataset", required=True, type=click.Path(), help="Dataset directory path"
)
@click.option(
    "--output", required=True, type=click.Path(), help="Output safetensors file path"
)
def steering(model: str, layer: str, dataset: str, output: str):
    """Extract steering vector using CAA method."""
    logger.info("Starting steering vector extraction")
    logger.info(f"Model: {model}")
    logger.info(f"Layer: {layer}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Output: {output}")

    console.print(f"[blue]Extracting steering vector...[/blue]")
    console.print(f"Model: {model}")
    console.print(f"Layer: {layer}")
    console.print(f"Dataset: {dataset}")
    console.print(f"Output: {output}")

    try:
        extractor = SteeringExtractor()
        extractor.extract_caa(model, layer, Path(dataset), Path(output))

        logger.success(f"Steering vector extracted successfully to {output}")
        console.print(
            f"[green]Steering vector extracted successfully to {output}[/green]"
        )
    except Exception as e:
        logger.error(f"Failed to extract steering vector: {e}")
        raise
