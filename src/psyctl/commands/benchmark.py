"""Benchmark commands for inventory testing."""

from pathlib import Path

import click
from rich.console import Console

from psyctl.core.inventory_tester import InventoryTester
from psyctl.core.logger import get_logger

console = Console()
logger = get_logger("benchmark")


@click.command()
@click.option("--model", required=True, help="Model name")
@click.option(
    "--steering-vector",
    required=True,
    type=click.Path(),
    help="Steering vector file path",
)
@click.option("--inventory", required=True, help="Inventory name (e.g., IPIP-NEO)")
def inventory(model: str, steering_vector: str, inventory: str):
    """Run inventory test to measure personality changes."""
    logger.info("Starting inventory test")
    logger.info(f"Model: {model}")
    logger.info(f"Steering vector: {steering_vector}")
    logger.info(f"Inventory: {inventory}")

    console.print(f"[blue]Running inventory test...[/blue]")
    console.print(f"Model: {model}")
    console.print(f"Steering vector: {steering_vector}")
    console.print(f"Inventory: {inventory}")

    try:
        tester = InventoryTester()
        results = tester.test_inventory(model, Path(steering_vector), inventory)

        logger.success("Inventory test completed")
        console.print(f"[green]Test results:[/green]")
        console.print(results)
    except Exception as e:
        logger.error(f"Failed to run inventory test: {e}")
        raise
