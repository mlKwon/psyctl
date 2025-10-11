"""Layer analysis commands."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from psyctl.core.layer_analyzer import LayerAnalyzer
from psyctl.core.logger import get_logger

console = Console()
logger = get_logger("layer")


@click.command()
@click.option(
    "--model", required=True, help="Model name (e.g., google/gemma-3-270m-it)"
)
@click.option(
    "--layer",
    multiple=True,
    help="Target layer paths (can be repeated, supports wildcards). "
    "Example: --layer 'model.layers[*].mlp' --layer 'model.layers[10:15].mlp'",
)
@click.option(
    "--layers",
    type=str,
    help="Comma-separated list of layer paths (supports wildcards). "
    "Example: --layers 'model.layers[*].mlp,model.layers[10:15].mlp.down_proj'",
)
@click.option(
    "--dataset",
    required=True,
    type=click.Path(exists=True),
    help="Dataset directory path",
)
@click.option(
    "--output",
    type=click.Path(),
    help="Output JSON file path (optional, skip if not provided)",
)
@click.option(
    "--method",
    type=str,
    default="svm",
    help="Analysis method: svm (default: svm)",
)
@click.option(
    "--top-k",
    type=int,
    default=5,
    help="Number of top layers to report (default: 5)",
)
@click.option(
    "--batch-size",
    type=int,
    help="Batch size for inference (default: from config)",
)
def analyze(
    model: str,
    layer: tuple[str],
    layers: str | None,
    dataset: str,
    output: str,
    method: str,
    top_k: int,
    batch_size: int | None,
):
    """
    Analyze layers to find optimal steering target layers.

    This command analyzes multiple layers to determine which ones provide
    the best separation between positive and neutral activations. It supports
    wildcard patterns for layer selection.

    Wildcard patterns supported:
    - [*] : all indices (e.g., "model.layers[*].mlp")
    - [start:end] : range (e.g., "model.layers[5:10].mlp")
    - [start:end:step] : range with step (e.g., "model.layers[0:20:2].mlp")
    - [start:] : from start to end (e.g., "model.layers[10:].mlp")
    - [:end] : from 0 to end (e.g., "model.layers[:5].mlp")

    Examples:

    \b
    # Analyze first 5 MLP layers
    psyctl layer.analyze \\
      --model "google/gemma-3-270m-it" \\
      --layers "model.layers[0:5].mlp" \\
      --dataset "./dataset/caa" \\
      --output "./results/layer_analysis.json"

    \b
    # Analyze all MLP layers
    psyctl layer.analyze \\
      --model "google/gemma-3-270m-it" \\
      --layers "model.layers[*].mlp" \\
      --dataset "./dataset/caa" \\
      --output "./results/layer_analysis.json" \\
      --top-k 10

    \b
    # Analyze specific layer types
    psyctl layer.analyze \\
      --model "meta-llama/Llama-3.2-3B-Instruct" \\
      --layers "model.layers[10:15].mlp.down_proj" \\
      --dataset "./dataset/caa" \\
      --output "./results/layer_analysis.json"

    \b
    # Multiple patterns with --layer flags
    psyctl layer.analyze \\
      --model "google/gemma-3-270m-it" \\
      --layer "model.layers[0:5].mlp" \\
      --layer "model.layers[10:15].mlp.down_proj" \\
      --dataset "./dataset/caa" \\
      --output "./results/layer_analysis.json"
    """
    logger.info("Starting layer analysis")
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
        layer_list.extend([
            layer_str.strip() for layer_str in layers.split(",") if layer_str.strip()
        ])

    # Validate that at least one layer is specified
    if not layer_list:
        raise click.UsageError(
            "No layers specified. Use --layer or --layers to specify target layers."
        )

    logger.info(f"Layer patterns ({len(layer_list)}): {layer_list}")

    console.print("[blue]Analyzing layers...[/blue]")
    console.print(f"Model: [cyan]{model}[/cyan]")
    console.print(f"Layer patterns ({len(layer_list)}):")
    for idx, layer_pattern in enumerate(layer_list, 1):
        console.print(f"  {idx}. [yellow]{layer_pattern}[/yellow]")
    console.print(f"Dataset: [cyan]{dataset}[/cyan]")
    if output:
        console.print(f"Output: [cyan]{output}[/cyan]")
    else:
        console.print("Output: [dim](not saving to file)[/dim]")
    console.print(f"Method: [cyan]{method}[/cyan]")
    console.print(f"Top-K: [cyan]{top_k}[/cyan]")
    if batch_size:
        console.print(f"Batch size: {batch_size}")

    try:
        analyzer = LayerAnalyzer()

        results = analyzer.analyze_layers(
            model_name=model,
            layers=layer_list,
            dataset_path=Path(dataset),
            output_path=Path(output) if output else None,
            batch_size=batch_size,
            method=method,
            top_k=top_k,
        )

        logger.info("Analysis completed successfully")
        console.print(f"\n[green]Analyzed {results['total_layers']} layers[/green]")

        # Display top results in a table
        table = Table(
            title=f"TOP {top_k} Layers (Best Separation)",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Layer", style="cyan")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Accuracy", justify="right")
        table.add_column("Margin", justify="right")

        for i, result in enumerate(results["rankings"][:top_k], 1):
            metrics = result["metrics"]
            score = metrics.get("score", 0)
            accuracy = metrics.get("accuracy", 0)
            margin = metrics.get("margin", 0)

            table.add_row(
                str(i),
                result["layer"],
                f"{score:.4f}",
                f"{accuracy:.4f}",
                f"{margin:.4f}",
            )

        console.print("\n")
        console.print(table)
        if output:
            console.print(f"\n[green]Results saved to: {output}[/green]")
        else:
            console.print("\n[dim]Results not saved (no output path specified)[/dim]")

    except Exception as e:
        logger.error(f"Failed to analyze layers: {e}")
        raise
