"""
Steering Dataset Generation Example: OpenRouter API

This example demonstrates how to generate steering datasets using OpenRouter API.
NO GPU required - uses cloud-based inference instead.

Requirements:
- .env file with HF_TOKEN and OPENROUTER_API_KEY
- Internet connection
- OpenRouter API credits

Advantages:
- No GPU required (works on any machine)
- Faster with parallel workers
- Access to larger/better models (e.g., Claude, GPT-4, Llama 405B)
- No model download needed

Disadvantages:
- API costs (varies by model)
- Requires internet connection
- Rate limits may apply
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from psyctl.core.dataset_builder import DatasetBuilder
from psyctl.core.logger import get_logger

# Initialize logger
logger = get_logger("dataset_generation_openrouter")

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

# Configuration
OPENROUTER_MODEL = "qwen/qwen3-next-80b-a3b-instruct"  # Default OpenRouter model
PERSONALITY = "Machiavellianism"  # Different personality from local example
SAMPLE_COUNT = 50  # More samples - faster with parallel workers
MAX_WORKERS = 4  # Parallel API calls for speed
RESULTS_DIR = Path("./results")
OUTPUT_DIR = RESULTS_DIR / "dataset_openrouter"

# Available OpenRouter models (examples)
AVAILABLE_MODELS = {
    "default": "qwen/qwen3-next-80b-a3b-instruct",  # Good balance
    "fast": "meta-llama/llama-3.3-70b-instruct",  # Fast and cheap
    "quality": "anthropic/claude-3.5-sonnet",  # Highest quality
    "large": "meta-llama/llama-3.1-405b-instruct",  # Largest open model
}

def main():
    """Generate steering dataset using OpenRouter API."""

    print("="*80)
    print("Steering Dataset Generation: OpenRouter API")
    print("="*80)
    print(f"Model: {OPENROUTER_MODEL} (cloud)")
    print(f"Personality: {PERSONALITY}")
    print(f"Samples: {SAMPLE_COUNT}")
    print(f"Workers: {MAX_WORKERS} (parallel)")
    print(f"Dataset: allenai/soda")
    print()
    print("NOTE: This will use OpenRouter API credits.")
    print("      Estimated cost: $0.01-0.05 for 50 samples (model dependent)")
    print()
    print("Available models:")
    for name, model_id in AVAILABLE_MODELS.items():
        current = " (SELECTED)" if model_id == OPENROUTER_MODEL else ""
        print(f"  {name}: {model_id}{current}")
    print("="*80)
    print()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Generate Steering Dataset with OpenRouter
    # =========================================================================
    print("Initializing DatasetBuilder with OpenRouter...")
    logger.info(f"Using OpenRouter model: {OPENROUTER_MODEL}")

    # Initialize with OpenRouter
    dataset_builder = DatasetBuilder(
        use_openrouter=True,
        openrouter_api_key=OPENROUTER_API_KEY,
        openrouter_max_workers=MAX_WORKERS  # Parallel processing
    )

    print(f"\nGenerating {SAMPLE_COUNT} steering dataset samples...")
    print(f"Using {MAX_WORKERS} parallel workers for speed...")
    print()

    try:
        dataset_file = dataset_builder.build_caa_dataset(
            model=OPENROUTER_MODEL,
            personality=PERSONALITY,
            output_dir=OUTPUT_DIR,
            limit_samples=SAMPLE_COUNT,
            dataset_name="allenai/soda",
            temperature=0.7,
            max_tokens=100
        )

        print(f"\n[SUCCESS] Dataset generated: {dataset_file}")
        logger.info(f"Dataset generated successfully: {dataset_file}")

        # Display samples
        print("\n" + "-"*80)
        print("DATASET SAMPLES (First 3 examples)")
        print("-"*80)
        import json
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                sample = json.loads(line)
                print(f"\n[Sample {i+1}]")
                print(f"Character: {sample['char_name']}")
                situation_preview = sample['situation'][:300]
                if len(sample['situation']) > 300:
                    situation_preview += "..."
                print(f"Situation:\n{situation_preview}")
                print(f"Positive: {sample['positive']}")
                print(f"Neutral: {sample['neutral']}")
        print("-"*80)

        # Summary statistics
        total_lines = sum(1 for _ in open(dataset_file, 'r', encoding='utf-8'))
        file_size = dataset_file.stat().st_size / 1024  # KB

        print("\n" + "="*80)
        print("GENERATION SUMMARY")
        print("="*80)
        print(f"Model: {OPENROUTER_MODEL}")
        print(f"Method: OpenRouter API")
        print(f"Parallel workers: {MAX_WORKERS}")
        print(f"Total samples: {total_lines}")
        print(f"File size: {file_size:.2f} KB")
        print(f"Output: {dataset_file}")

        # Show OpenRouter usage if available
        if hasattr(dataset_builder, 'openrouter_client') and dataset_builder.openrouter_client:
            total_requests = dataset_builder.openrouter_client.get_total_requests()
            total_cost = dataset_builder.openrouter_client.get_total_cost()
            print(f"\nOpenRouter Usage:")
            print(f"  Total requests: {total_requests}")
            print(f"  Total cost: ${total_cost:.6f}")

        print()
        print("Next steps:")
        print("1. Extract steering vector using this dataset")
        print("2. Apply steering to see personality changes")
        print()
        print("Tip: Try different models by changing OPENROUTER_MODEL variable:")
        print("     - Fast & cheap: meta-llama/llama-3.3-70b-instruct")
        print("     - High quality: anthropic/claude-3.5-sonnet")
        print("="*80)

    except Exception as e:
        logger.error(f"Failed to generate dataset: {e}")
        print(f"\n[ERROR] Dataset generation failed: {e}")
        print("\nTroubleshooting:")
        print("- Check OPENROUTER_API_KEY in .env file")
        print("- Verify you have API credits: https://openrouter.ai/credits")
        print("- Try reducing MAX_WORKERS if hitting rate limits")
        print("- Check model availability: https://openrouter.ai/models")
        raise

if __name__ == "__main__":
    main()
