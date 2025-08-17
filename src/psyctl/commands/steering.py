"""Steering experiment commands."""

from pathlib import Path

import click
from rich.console import Console

from psyctl.core.logger import get_logger
from psyctl.core.steering_applier import SteeringApplier

console = Console()
logger = get_logger("steering")


@click.command()
@click.option("--model", required=True, help="Model name")
@click.option(
    "--steering-vector",
    required=True,
    type=click.Path(),
    help="Steering vector file path",
)
@click.option("--input-text", required=True, help="Input text for generation")
def apply(model: str, steering_vector: str, input_text: str):
    """Apply steering vector and generate text."""
    logger.info("Starting steering vector application")
    logger.info(f"Model: {model}")
    logger.info(f"Steering vector: {steering_vector}")
    logger.info(f"Input text: {input_text}")

    console.print(f"[blue]Applying steering vector...[/blue]")
    console.print(f"Model: {model}")
    console.print(f"Steering vector: {steering_vector}")
    console.print(f"Input text: {input_text}")

    try:
        applier = SteeringApplier()
        result = applier.apply_steering(model, Path(steering_vector), input_text)

        logger.success("Text generation completed")
        console.print(f"[green]Generated text:[/green]")
        console.print(result)
    except Exception as e:
        logger.error(f"Failed to apply steering vector: {e}")
        raise
