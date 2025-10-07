"""Dataset generation commands."""

from pathlib import Path

import click
from rich.console import Console

from psyctl.core.dataset_builder import DatasetBuilder
from psyctl.core.logger import get_logger

console = Console()
logger = get_logger("dataset")


@click.command()
@click.option("--model", required=False, help="Model name (e.g., google/gemma-3-27b-it)")
@click.option(
    "--personality",
    required=True,
    help="Personality traits (e.g., Extroversion, Machiavellism)",
)
@click.option(
    "--output", required=True, type=click.Path(), help="Output directory path"
)
@click.option(
    "--limit-samples",
    required=False,
    type=int,
    default=0,
    help="Maximum number of samples to generate",
)
@click.option(
    "--dataset",
    required=False,
    default="allenai/soda",
    help="Hugging Face dataset name (e.g., allenai/soda, username/custom-dataset)",
)
@click.option(
    "--openrouter-api-key",
    required=False,
    help="OpenRouter API key (format: sk-or-xxxx). If provided, uses OpenRouter instead of local model.",
)
@click.option(
    "--openrouter-model",
    required=False,
    default="qwen/qwen3-next-80b-a3b-instruct",
    help="OpenRouter model identifier (default: qwen/qwen3-next-80b-a3b-instruct)",
)
@click.option(
    "--openrouter-max-workers",
    required=False,
    type=int,
    default=1,
    help="Number of parallel workers for OpenRouter API (1 = sequential, higher = parallel)",
)
@click.option(
    "--caa-question-template",
    required=False,
    type=click.Path(exists=True),
    help="Path to custom Jinja2 template for CAA questions (.j2 file)",
)
@click.option(
    "--roleplay-prompt-template",
    required=False,
    type=click.Path(exists=True),
    help="Path to custom Jinja2 template for roleplay prompts (.j2 file)",
)
def build_caa(
    model: str,
    personality: str,
    output: str,
    limit_samples: int,
    dataset: str,
    openrouter_api_key: str,
    openrouter_model: str,
    openrouter_max_workers: int,
    caa_question_template: str,
    roleplay_prompt_template: str,
):
    """Build CAA dataset for steering vector extraction."""
    # Determine if using OpenRouter or local model
    use_openrouter = bool(openrouter_api_key)

    # Validate configuration
    if use_openrouter:
        logger.info("Using OpenRouter API mode")
        console.print("[yellow]Using OpenRouter API mode[/yellow]")
        if not model:
            model = "openrouter"  # Placeholder when using OpenRouter
    else:
        if not model:
            logger.error("--model is required when not using OpenRouter")
            console.print("[red]Error: --model is required when not using --openrouter-api-key[/red]")
            raise click.BadParameter("--model is required when not using OpenRouter")
        logger.info("Using local model mode")

    logger.info("Starting CAA dataset build")
    logger.info(f"Model: {model}")
    logger.info(f"Personality: {personality}")
    logger.info(f"Output: {output}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Limit samples: {limit_samples}")

    console.print(f"[blue]Building CAA dataset...[/blue]")
    if use_openrouter:
        console.print(f"OpenRouter Model: {openrouter_model}")
        console.print(f"OpenRouter Workers: {openrouter_max_workers}")
    else:
        console.print(f"Local Model: {model}")
    console.print(f"Personality: {personality}")
    console.print(f"Output: {output}")
    console.print(f"Dataset: {dataset}")
    console.print(f"Limit samples: {limit_samples}")

    try:
        builder = DatasetBuilder(
            use_openrouter=use_openrouter,
            openrouter_api_key=openrouter_api_key,
            openrouter_model=openrouter_model,
            openrouter_max_workers=openrouter_max_workers,
            caa_question_template=caa_question_template,
            roleplay_prompt_template=roleplay_prompt_template,
        )
        output_file = builder.build_caa_dataset(model, personality, Path(output), limit_samples, dataset)

        logger.info(f"Dataset built successfully: {output_file}")
        console.print(f"[green]Dataset built successfully: {output_file}[/green]")
    except Exception as e:
        logger.error(f"Failed to build dataset: {e}")
        raise
