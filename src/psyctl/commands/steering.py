"""Steering experiment commands."""

from pathlib import Path

import click
from rich.console import Console

from psyctl.core.logger import get_logger
from psyctl.core.steering_applier import SteeringApplier

console = Console()
logger = get_logger("steering")


@click.command()
@click.option("--model", required=True, help="Model name or HuggingFace identifier")
@click.option(
    "--steering-vector",
    required=True,
    type=click.Path(exists=True),
    help="Path to steering vector file (.safetensors)",
)
@click.option("--input-text", required=True, help="Input text for generation")
@click.option(
    "--strength",
    default=1.0,
    type=float,
    help="Steering strength multiplier (default: 1.0)",
)
@click.option(
    "--max-tokens",
    default=200,
    type=int,
    help="Maximum number of tokens to generate (default: 200)",
)
@click.option(
    "--temperature",
    default=1.0,
    type=float,
    help="Sampling temperature, 0 for greedy (default: 1.0)",
)
@click.option(
    "--top-p",
    default=0.9,
    type=float,
    help="Top-p (nucleus) sampling parameter (default: 0.9)",
)
@click.option(
    "--top-k",
    default=50,
    type=int,
    help="Top-k sampling parameter (default: 50)",
)
@click.option(
    "--orthogonal",
    is_flag=True,
    help="Use orthogonalized addition method",
)
def apply(
    model: str,
    steering_vector: str,
    input_text: str,
    strength: float,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    orthogonal: bool,
):
    """
    Apply steering vector and generate text.

    This command loads a pre-extracted steering vector and applies it to the model
    during text generation. The steering vector influences the model's personality
    or behavior according to the training data used during extraction.

    Examples:

        Basic usage (simple addition):

            psyctl steering --model google/gemma-3-270m-it --steering-vector ./vector.safetensors --input-text "hello world"

        With custom strength:

            psyctl steering --model google/gemma-3-270m-it --steering-vector ./vector.safetensors --input-text "Tell me about yourself" --strength 1.5

        Using orthogonalized addition:

            psyctl steering --model google/gemma-3-270m-it --steering-vector ./vector.safetensors --input-text "hello" --orthogonal --strength 2.0
    """
    logger.info("Starting steering vector application")
    logger.info(f"Model: {model}")
    logger.info(f"Steering vector: {steering_vector}")
    logger.info(f"Input text: {input_text}")
    logger.info(f"Strength: {strength}")

    console.print("[blue]Applying steering vector...[/blue]")
    console.print(f"[cyan]Model:[/cyan] {model}")
    console.print(f"[cyan]Steering vector:[/cyan] {steering_vector}")
    console.print(f"[cyan]Strength:[/cyan] {strength}")
    console.print(f"[cyan]Temperature:[/cyan] {temperature}")
    console.print(f"[cyan]Max tokens:[/cyan] {max_tokens}")
    if orthogonal:
        console.print("[cyan]Method:[/cyan] Orthogonalized addition")
    else:
        console.print("[cyan]Method:[/cyan] Simple addition")
    console.print()

    try:
        applier = SteeringApplier()
        result = applier.apply_steering(
            model_name=model,
            steering_vector_path=Path(steering_vector),
            input_text=input_text,
            strength=strength,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            orthogonal=orthogonal,
        )

        logger.info("Text generation completed")
        console.print("[green]Generated text:[/green]")
        console.print()
        console.print(result)
        console.print()

    except Exception as e:
        logger.error(f"Failed to apply steering vector: {e}")
        console.print(f"[red]Error: {e}[/red]")
        raise
