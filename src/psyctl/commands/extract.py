"""Steering vector extraction commands."""

from pathlib import Path
from typing import Optional, Tuple

import click
from rich.console import Console

from psyctl.core.logger import get_logger
from psyctl.core.steering_extractor import SteeringExtractor

console = Console()
logger = get_logger("extract")


@click.command()
@click.option("--model", required=True, help="Model name (e.g., meta-llama/Llama-3.2-3B-Instruct)")
@click.option(
    "--layer",
    multiple=True,
    help="Target layer path (can be repeated for multi-layer extraction). "
    "Example: --layer 'model.layers[13].mlp.down_proj' --layer 'model.layers[14].mlp.down_proj'",
)
@click.option(
    "--layers",
    type=str,
    help="Comma-separated list of layer paths. "
    "Example: --layers 'model.layers[13].mlp.down_proj,model.layers[14].mlp.down_proj'",
)
@click.option(
    "--dataset", required=True, type=click.Path(exists=True), help="Dataset directory path"
)
@click.option(
    "--output", required=True, type=click.Path(), help="Output safetensors file path"
)
@click.option(
    "--batch-size",
    type=int,
    help="Batch size for inference (default: from config)",
)
@click.option(
    "--normalize",
    is_flag=True,
    help="Normalize steering vectors to unit length",
)
@click.option(
    "--method",
    type=str,
    default="mean_contrastive",
    help="Extraction method (default: mean_contrastive)",
)
def steering(
    model: str,
    layer: Tuple[str],
    layers: Optional[str],
    dataset: str,
    output: str,
    batch_size: Optional[int],
    normalize: bool,
    method: str,
):
    """
    Extract steering vectors using CAA method.

    Supports single or multi-layer extraction. Specify layers using either:
    - Multiple --layer flags: --layer "model.layers[13].mlp.down_proj" --layer "model.layers[14].mlp.down_proj"
    - Comma-separated --layers: --layers "model.layers[13].mlp.down_proj,model.layers[14].mlp.down_proj"

    Examples:

    \b
    # Single layer extraction
    psyctl extract.steering \\
      --model "meta-llama/Llama-3.2-3B-Instruct" \\
      --layer "model.layers[13].mlp.down_proj" \\
      --dataset "./dataset/caa" \\
      --output "./steering_vector/out.safetensors"

    \b
    # Multi-layer extraction with repeated --layer
    psyctl extract.steering \\
      --model "meta-llama/Llama-3.2-3B-Instruct" \\
      --layer "model.layers[13].mlp.down_proj" \\
      --layer "model.layers[14].mlp.down_proj" \\
      --dataset "./dataset/caa" \\
      --output "./steering_vector/out.safetensors"

    \b
    # Multi-layer extraction with comma-separated --layers
    psyctl extract.steering \\
      --model "meta-llama/Llama-3.2-3B-Instruct" \\
      --layers "model.layers[13].mlp.down_proj,model.layers[14].mlp.down_proj" \\
      --dataset "./dataset/caa" \\
      --output "./steering_vector/out.safetensors"
    """
    logger.info("Starting steering vector extraction")
    logger.info(f"Model: {model}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Output: {output}")
    logger.info(f"Method: {method}")

    # Parse layer arguments
    layer_list = []

    # Add layers from --layer flags (multiple)
    if layer:
        layer_list.extend(layer)

    # Add layers from --layers (comma-separated)
    if layers:
        layer_list.extend([l.strip() for l in layers.split(",") if l.strip()])

    # Validate that at least one layer is specified
    if not layer_list:
        raise click.UsageError(
            "No layers specified. Use --layer or --layers to specify target layers."
        )

    logger.info(f"Target layers ({len(layer_list)}): {layer_list}")

    console.print(f"[blue]Extracting steering vectors...[/blue]")
    console.print(f"Model: [cyan]{model}[/cyan]")
    console.print(f"Layers ({len(layer_list)}):")
    for idx, layer_name in enumerate(layer_list, 1):
        console.print(f"  {idx}. [yellow]{layer_name}[/yellow]")
    console.print(f"Dataset: [cyan]{dataset}[/cyan]")
    console.print(f"Output: [cyan]{output}[/cyan]")
    console.print(f"Method: [cyan]{method}[/cyan]")
    if normalize:
        console.print("[yellow]Normalization: Enabled[/yellow]")
    if batch_size:
        console.print(f"Batch size: {batch_size}")

    try:
        extractor = SteeringExtractor()
        vectors = extractor.extract_caa(
            model_name=model,
            layers=layer_list,
            dataset_path=Path(dataset),
            output_path=Path(output),
            batch_size=batch_size,
            normalize=normalize,
            method=method,
        )

        logger.info(f"Steering vectors extracted successfully to {output}")
        console.print(
            f"[green]Extracted {len(vectors)} steering vectors to {output}[/green]"
        )

        # Display vector info
        console.print("\n[bold]Extracted Vectors:[/bold]")
        for layer_name, vector in vectors.items():
            norm = vector.norm().item()
            console.print(
                f"  - [cyan]{layer_name}[/cyan]: shape={list(vector.shape)}, norm={norm:.4f}"
            )

    except Exception as e:
        logger.error(f"Failed to extract steering vector: {e}")
        raise
