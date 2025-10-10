#!/usr/bin/env python3
"""
Example: Layer Analysis for Finding Optimal Steering Target Layers

This example demonstrates how to analyze multiple layers to determine
which ones provide the best separation between positive and neutral activations.

Steps:
1. Generate or load a steering dataset
2. Analyze layers using wildcard patterns
3. Review results to select optimal target layers
4. Use top layers for steering vector extraction

Usage:
    python examples/layer_analysis_example.py
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from psyctl.core.layer_analyzer import LayerAnalyzer

console = Console()

# Configuration
MODEL_NAME = "google/gemma-3-270m-it"  # Small model for testing
DATASET_PATH = Path(
    "./results/korean_extroversion_caa/caa_dataset_20251009_193430.jsonl"
)
OUTPUT_PATH = Path("./results/layer_analysis_example.json")

# Layer patterns to analyze (supports wildcards)
LAYER_PATTERNS = [
    "model.layers[*].mlp",  # All MLP layers
    # Or specific ranges:
    # "model.layers[5:15].mlp",  # Layers 5-14
    # "model.layers[::2].mlp",   # Every other layer
]

console.print("\n[bold cyan]Layer Analysis Example[/bold cyan]")
console.print("=" * 70)
console.print(f"Model: [yellow]{MODEL_NAME}[/yellow]")
console.print(f"Dataset: [yellow]{DATASET_PATH}[/yellow]")
console.print(f"Patterns: [yellow]{LAYER_PATTERNS}[/yellow]")
console.print("=" * 70)

# Check if dataset exists
if not DATASET_PATH.exists():
    console.print(f"\n[red]Error: Dataset not found at {DATASET_PATH}[/red]")
    console.print("\n[yellow]Please run the following command first:[/yellow]")
    console.print("  psyctl dataset.build.steer \\")
    console.print(f'    --model "{MODEL_NAME}" \\')
    console.print('    --personality "Extroversion" \\')
    console.print(f'    --output "{DATASET_PATH.parent}" \\')
    console.print("    --num-samples 100")
    sys.exit(1)

try:
    # Step 1: Create analyzer
    analyzer = LayerAnalyzer()

    # Step 2: Analyze layers
    console.print("\n[cyan]Analyzing layers...[/cyan]")
    results = analyzer.analyze_layers(
        model_name=MODEL_NAME,
        layers=LAYER_PATTERNS,
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_PATH,
        method="svm",  # SVM-based separation analysis
        top_k=5,  # Report top 5 layers
        batch_size=8,
    )

    # Step 3: Display results in a table
    console.print(f"\n[green]Analyzed {results['total_layers']} layers[/green]")

    table = Table(
        title="TOP 5 Layers (Best Separation)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Layer", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Accuracy", justify="right")
    table.add_column("Margin", justify="right")

    for i, result in enumerate(results["rankings"][:5], 1):
        metrics = result["metrics"]
        score = metrics.get("score", 0)
        accuracy = metrics.get("accuracy", 0)
        margin = metrics.get("margin", 0)

        table.add_row(
            str(i), result["layer"], f"{score:.4f}", f"{accuracy:.4f}", f"{margin:.4f}"
        )

    console.print("\n")
    console.print(table)

    console.print("\n[bold]Recommended Layers for Steering:[/bold]")
    for layer in results["top_k_layers"]:
        console.print(f"  [green]*[/green] {layer}")

    console.print(f"\n[green]Full results saved to: {OUTPUT_PATH}[/green]")

    # Step 4: Show next steps
    console.print("\n" + "=" * 70)
    console.print("[bold]Next Steps:[/bold]")
    console.print("=" * 70)
    console.print(
        "\n[yellow]1. Extract steering vector from the top-ranked layer:[/yellow]"
    )
    console.print("   psyctl extract.steering \\")
    console.print(f'     --model "{MODEL_NAME}" \\')
    console.print(f'     --layer "{results["top_k_layers"][0]}" \\')
    console.print(f'     --dataset "{DATASET_PATH}" \\')
    console.print('     --output "./steering_vector/best_layer.safetensors"')

    console.print("\n[yellow]2. Or extract from multiple top layers:[/yellow]")
    console.print("   psyctl extract.steering \\")
    console.print(f'     --model "{MODEL_NAME}" \\')
    for layer in results["top_k_layers"][:3]:
        console.print(f'     --layer "{layer}" \\')
    console.print(f'     --dataset "{DATASET_PATH}" \\')
    console.print('     --output "./steering_vector/multi_layer.safetensors"')

    console.print("\n[yellow]3. Apply steering to test:[/yellow]")
    console.print("   psyctl steering \\")
    console.print(f'     --model "{MODEL_NAME}" \\')
    console.print(
        '     --steering-vector "./steering_vector/best_layer.safetensors" \\'
    )
    console.print('     --input-text "Hello, how are you?"')

    console.print("\n" + "=" * 70 + "\n")

except FileNotFoundError as e:
    console.print(f"\n[red]Error: {e}[/red]")
    sys.exit(1)
except Exception as e:
    console.print(f"\n[red]Error during analysis: {e}[/red]")
    import traceback

    traceback.print_exc()
    sys.exit(1)
