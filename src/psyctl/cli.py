#!/usr/bin/env python3
"""Main CLI entry point for psyctl."""

import click
import torch
from dotenv import load_dotenv
from rich.console import Console
from rich.traceback import install

from psyctl.commands import benchmark, dataset, extract, steering
from psyctl.config.settings import Settings
from psyctl.core.logger import get_logger, setup_logging

# Disable PyTorch compiler to avoid Triton issues
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

load_dotenv(override=True)

# Rich 설정
install(show_locals=True)
console = Console()

# Setup logging
settings = Settings()
setup_logging(settings)
logger = get_logger("cli")


@click.group()
@click.version_option(prog_name="psyctl")
def main():
    """PSYCTL - LLM Personality Steering Tool."""
    logger.info("PSYCTL CLI started")
    pass


# 명령어 등록
main.add_command(dataset.build_caa, name="dataset.build.caa")
main.add_command(extract.steering, name="extract.steering")
main.add_command(steering.apply, name="steering")
main.add_command(benchmark.inventory, name="benchmark")

if __name__ == "__main__":
    main()
