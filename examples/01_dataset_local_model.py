"""
Steering Dataset Generation Example: Local Model

This example demonstrates how to generate steering datasets using a local model.
NO OpenRouter API required - uses your local GPU instead.

Requirements:
- .env file with HF_TOKEN
- GPU with sufficient VRAM (recommended: 8GB+)
- Local disk space for model cache

Advantages:
- No API costs
- Full control over model and parameters
- Works offline after initial model download

Disadvantages:
- Requires GPU
- Slower than OpenRouter with parallel workers
- Model download required (first time)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from psyctl.core.dataset_builder import DatasetBuilder
from psyctl.core.logger import get_logger

# Initialize logger
logger = get_logger("dataset_generation_local")

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

# Configuration
LOCAL_MODEL = "google/gemma-3-270m-it"  # Small model for local generation
PERSONALITY = "Extroversion"
SAMPLE_COUNT = 20  # More samples for better steering quality
RESULTS_DIR = Path("./results")
OUTPUT_DIR = RESULTS_DIR / "dataset_local"


def main():
    """Generate steering dataset using local model."""

    print("=" * 80)
    print("Steering Dataset Generation: Local Model")
    print("=" * 80)
    print(f"Model: {LOCAL_MODEL} (local)")
    print(f"Personality: {PERSONALITY}")
    print(f"Samples: {SAMPLE_COUNT}")
    print("Dataset: allenai/soda")
    print()
    print("NOTE: This will download the model if not cached (~540MB)")
    print("      and requires GPU for reasonable speed.")
    print("=" * 80)
    print()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Generate Steering Dataset with Local Model
    # =========================================================================
    print("Initializing DatasetBuilder with local model...")
    logger.info(f"Using local model: {LOCAL_MODEL}")

    # Initialize without OpenRouter
    dataset_builder = DatasetBuilder(
        use_openrouter=False  # Use local model
    )

    print(f"\nGenerating {SAMPLE_COUNT} steering dataset samples...")
    print("This may take a few minutes depending on your GPU...")
    print()

    try:
        dataset_file = dataset_builder.build_caa_dataset(
            model=LOCAL_MODEL,
            personality=PERSONALITY,
            output_dir=OUTPUT_DIR,
            limit_samples=SAMPLE_COUNT,
            dataset_name="allenai/soda",
            temperature=0.7,
            max_tokens=100,
        )

        print(f"\n[SUCCESS] Dataset generated: {dataset_file}")
        logger.info(f"Dataset generated successfully: {dataset_file}")

        # Display samples
        print("\n" + "-" * 80)
        print("DATASET SAMPLES (First 3 examples)")
        print("-" * 80)
        import json

        with Path(dataset_file).open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                sample = json.loads(line)
                print(f"\n[Sample {i + 1}]")
                print(f"Character: {sample['char_name']}")
                # Show more of the situation for local generation
                situation_preview = sample["situation"][:300]
                if len(sample["situation"]) > 300:
                    situation_preview += "..."
                print(f"Situation:\n{situation_preview}")
                print(f"Positive: {sample['positive']}")
                print(f"Neutral: {sample['neutral']}")
        print("-" * 80)

        # Summary statistics
        with Path(dataset_file).open(encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
        file_size = dataset_file.stat().st_size / 1024  # KB

        print("\n" + "=" * 80)
        print("GENERATION SUMMARY")
        print("=" * 80)
        print(f"Model: {LOCAL_MODEL}")
        print("Method: Local GPU inference")
        print(f"Total samples: {total_lines}")
        print(f"File size: {file_size:.2f} KB")
        print(f"Output: {dataset_file}")
        print()
        print("Next steps:")
        print("1. Extract steering vector using this dataset")
        print("2. Apply steering to see personality changes")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Failed to generate dataset: {e}")
        print(f"\n[ERROR] Dataset generation failed: {e}")
        print("\nTroubleshooting:")
        print("- Check if GPU is available: torch.cuda.is_available()")
        print("- Try a smaller sample count (e.g., --limit-samples 10)")
        print("- Ensure sufficient GPU VRAM (recommended: 8GB+)")
        raise


if __name__ == "__main__":
    main()
