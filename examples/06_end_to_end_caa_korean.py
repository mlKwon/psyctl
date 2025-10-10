"""
PSYCTL End-to-End Example: Korean Rudeness with Mean Diff Method

This script demonstrates the complete workflow using mean_diff extraction method:
1. Generate steering dataset using OpenRouter API with Korean dialogue data
2. Extract steering vector using mean_diff method (Mean Difference from CAA paper - faster than BiPO)
3. Apply steering using CAA (Contrastive Activation Addition) to test personality transformation

Extraction Methods:
- mean_diff: Computes mean difference between positive/neutral activations (fast, no training)
- bipo: Optimizes steering vector through gradient descent (slower, potentially better)

Requirements:
- .env file with HF_TOKEN and OPENROUTER_API_KEY
- Internet connection for OpenRouter API
- ~2GB disk space for model cache
"""

import argparse
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

# Import PSYCTL components
from psyctl.core.dataset_builder import DatasetBuilder
from psyctl.core.logger import get_logger
from psyctl.core.steering_applier import SteeringApplier
from psyctl.core.steering_extractor import SteeringExtractor
from psyctl.models.llm_loader import LLMLoader

# Initialize logger
logger = get_logger("end_to_end_caa")

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Validate API keys
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

# Configuration
DATASET_MODEL = "moonshotai/kimi-k2-0905"  # OpenRouter model for dataset generation
STEERING_MODEL = (
    "google/gemma-3-270m-it"  # Local model for steering extraction and application
)
PERSONALITY = "매우 무례한"  # Very rude personality
SAMPLE_COUNT = 10  # Small sample for quick testing
RESULTS_DIR = Path("./results")
DATASET_OUTPUT = RESULTS_DIR / "korean_rudeness_caa"
STEERING_VECTOR_PATH = RESULTS_DIR / "korean_rudeness_caa_steering.safetensors"

# Custom Korean template for roleplay prompts
ROLEPLAY_TEMPLATE = """# 개요
이것은 롤플레이 세션입니다.
당신(어시스턴트 또는 모델)의 역할은 {{ char_name }}입니다.
사용자의 역할은 {{ user_name }}입니다.
{{ char_name }}의 짧은 반응을 한 문장으로 작성하세요.

# {{ char_name }}에 대하여
{{ p2 }}

# 상황
{{ situation }}
"""


def main():
    """Execute the complete end-to-end workflow with mean_diff extraction method."""
    parser = argparse.ArgumentParser(description="PSYCTL End-to-End Mean Diff Example")
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip dataset generation and use existing dataset",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to existing dataset file (required with --skip-dataset)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("PSYCTL End-to-End Example: Mean Diff Extraction Method")
    print("=" * 80)
    print()

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STEP 1: Generate Steering Dataset with OpenRouter
    # =========================================================================
    if args.skip_dataset:
        if not args.dataset_path:
            raise ValueError("--dataset-path is required when using --skip-dataset")
        dataset_file = Path(args.dataset_path)
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        print("\n" + "=" * 80)
        print("STEP 1: Using Existing Dataset")
        print("=" * 80)
        print(f"Dataset: {dataset_file}")
        print()
    else:
        print("\n" + "=" * 80)
        print("STEP 1: Generating Steering Dataset")
        print("=" * 80)
        print(f"Model: {DATASET_MODEL}")
        print(f"Personality: {PERSONALITY}")
        print("Dataset: CaveduckAI/simplified_soda_kr (Korean dialogues)")
        print(f"Samples: {SAMPLE_COUNT}")
        print()

        logger.info("Initializing DatasetBuilder with OpenRouter")
        dataset_builder = DatasetBuilder(
            use_openrouter=True,
            openrouter_api_key=OPENROUTER_API_KEY,
            openrouter_max_workers=2,  # Parallel processing
        )

        # Set custom Korean template
        logger.info("Setting custom Korean roleplay template")
        dataset_builder.set_roleplay_prompt_template(ROLEPLAY_TEMPLATE)

        # Build steering dataset
        logger.info("Starting steering dataset generation")
        try:
            dataset_file = dataset_builder.build_caa_dataset(
                model=DATASET_MODEL,
                personality=PERSONALITY,
                output_dir=DATASET_OUTPUT,
                limit_samples=SAMPLE_COUNT,
                dataset_name="CaveduckAI/simplified_soda_kr",
                temperature=0.7,
                max_tokens=100,
            )
            print(f"\n[SUCCESS] Steering dataset generated: {dataset_file}")
            logger.info(f"Steering dataset generated successfully: {dataset_file}")

            # Display sample data from generated dataset
            print("\n" + "-" * 80)
            print("DATASET SAMPLES (First 2 examples)")
            print("-" * 80)
            import json

            with Path(dataset_file).open(encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 2:  # Show only first 2 samples
                        break
                    sample = json.loads(line)
                    print(f"\n[Sample {i + 1}]")
                    print(f"Character: {sample['char_name']}")
                    print(f"Situation: {sample['situation'][:200]}...")
                    print(f"Positive answer: {sample['positive']}")
                    print(f"Neutral answer: {sample['neutral']}")
            print("-" * 80)

        except Exception as e:
            logger.error(f"Failed to generate steering dataset: {e}")
            raise

    # =========================================================================
    # STEP 2: Extract Steering Vector using Mean Diff Method
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Extracting Steering Vector")
    print("=" * 80)
    print(f"Model: {STEERING_MODEL}")
    print(f"Dataset: {dataset_file}")
    print("Method: mean_diff (Mean Difference)")
    print("Note: mean_diff is faster than BiPO (no training required)")
    print()

    logger.info("Initializing SteeringExtractor")
    extractor = SteeringExtractor()

    # Extract steering vector using mean_diff method
    logger.info("Starting steering vector extraction using mean_diff method")
    try:
        # Determine target layers - use middle layer for gemma-3-270m-it
        target_layers = [
            "model.layers.9.mlp.down_proj"
        ]  # Middle layer for gemma-3-270m-it (18 layers)

        extractor.extract_steering_vector(
            model_name=STEERING_MODEL,
            layers=target_layers,
            dataset_path=dataset_file,
            output_path=STEERING_VECTOR_PATH,
            method="mean_diff",  # Mean Difference extraction
            normalize=True,
        )
        print(f"\n[SUCCESS] Steering vector extracted: {STEERING_VECTOR_PATH}")
        logger.info(f"Steering vector extracted successfully: {STEERING_VECTOR_PATH}")
    except Exception as e:
        logger.error(f"Failed to extract steering vector: {e}")
        raise

    # =========================================================================
    # STEP 3: Apply Steering and Test
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Applying Steering and Testing")
    print("=" * 80)
    print(f"Model: {STEERING_MODEL}")
    print(f"Steering Vector: {STEERING_VECTOR_PATH}")
    print("Test Input: '안녕하세요' (Hello)")
    print()

    logger.info("Initializing SteeringApplier")
    applier = SteeringApplier()

    # Test input
    test_input = "안녕하세요"

    # Generate response WITHOUT steering (using a simple baseline generation)
    print("Generating response WITHOUT steering...")
    logger.info("Generating baseline response (no steering)")
    try:
        loader = LLMLoader()
        model, tokenizer = loader.load_model(STEERING_MODEL)

        # Prepare prompt
        messages = [{"role": "user", "content": test_input}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=100, temperature=0.7, top_p=0.9, do_sample=True
            )

        response_baseline = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        print(f"\n[BASELINE] Input: {test_input}")
        print(f"[BASELINE] Response: {response_baseline}")

        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        logger.error(f"Failed to generate baseline response: {e}")
        raise

    # Generate response WITH steering
    print("\nGenerating response WITH steering...")
    logger.info("Generating steered response (rudeness personality)")
    try:
        response_steered = applier.apply_steering(
            model_name=STEERING_MODEL,
            steering_vector_path=STEERING_VECTOR_PATH,
            input_text=test_input,
            max_new_tokens=100,
            strength=1.5,  # Higher strength for clear effect
            temperature=0.7,
        )
        print(f"\n[STEERED] Input: {test_input}")
        print(f"[STEERED] Response: {response_steered}")
    except Exception as e:
        logger.error(f"Failed to generate steered response: {e}")
        raise

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"[SUCCESS] Steering Dataset: {dataset_file}")
    print(f"[SUCCESS] Steering Vector: {STEERING_VECTOR_PATH}")
    print(f"\n[SUCCESS] Baseline Response: {response_baseline}")
    print(f"[SUCCESS] Steered Response: {response_steered}")
    print("\nEnd-to-end workflow completed successfully!")
    print(
        "Extraction: mean_diff (MD) | Application: CAA (Contrastive Activation Addition)"
    )
    print("=" * 80)

    logger.info("End-to-end workflow completed successfully")


if __name__ == "__main__":
    main()
