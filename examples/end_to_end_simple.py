"""
PSYCTL End-to-End Simple Example: Basic Personality Steering

This is the simplest example demonstrating the complete PSYCTL workflow:
1. Generate CAA dataset using OpenRouter API (default templates, English)
2. Extract CAA steering vector (fast mean contrastive method)
3. Apply steering to test the personality transformation

Simplifications:
- Uses default English templates (no customization)
- Uses allenai/soda dataset (default)
- Uses CAA method (faster than BiPO)
- Simple personality: "Extroversion"

Requirements:
- .env file with HF_TOKEN and OPENROUTER_API_KEY
- Internet connection for OpenRouter API
- ~2GB disk space for model cache
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
import torch

# Import PSYCTL components
from psyctl.core.dataset_builder import DatasetBuilder
from psyctl.core.steering_extractor import SteeringExtractor
from psyctl.core.steering_applier import SteeringApplier
from psyctl.models.llm_loader import LLMLoader
from psyctl.core.logger import get_logger

# Initialize logger
logger = get_logger("end_to_end_simple")

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Validate API keys
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

# Configuration (all defaults, as simple as possible)
DATASET_MODEL = "qwen/qwen3-next-80b-a3b-instruct"  # Default OpenRouter model
STEERING_MODEL = "google/gemma-3-270m-it"  # Small local model for testing
PERSONALITY = "Extroversion"  # Simple personality trait
SAMPLE_COUNT = 10  # Small sample for quick testing
RESULTS_DIR = Path("./results")
DATASET_OUTPUT = RESULTS_DIR / "simple_example"
STEERING_VECTOR_PATH = RESULTS_DIR / "simple_steering.safetensors"

def main():
    """Execute the simplest end-to-end workflow."""
    parser = argparse.ArgumentParser(description="PSYCTL Simple End-to-End Example")
    parser.add_argument("--skip-dataset", action="store_true",
                        help="Skip dataset generation and use existing dataset")
    parser.add_argument("--dataset-path", type=str,
                        help="Path to existing dataset file (required with --skip-dataset)")
    args = parser.parse_args()

    print("="*80)
    print("PSYCTL Simple End-to-End Example")
    print("="*80)
    print("This example uses:")
    print("- Default English templates")
    print("- allenai/soda dataset")
    print("- CAA extraction method (fast)")
    print("- Extroversion personality")
    print()

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STEP 1: Generate CAA Dataset (Default Settings)
    # =========================================================================
    if args.skip_dataset:
        if not args.dataset_path:
            raise ValueError("--dataset-path is required when using --skip-dataset")
        dataset_file = Path(args.dataset_path)
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        print("\n" + "="*80)
        print("STEP 1: Using Existing Dataset")
        print("="*80)
        print(f"Dataset: {dataset_file}")
        print()
    else:
        print("\n" + "="*80)
        print("STEP 1: Generating CAA Dataset")
        print("="*80)
        print(f"Model: {DATASET_MODEL}")
        print(f"Personality: {PERSONALITY}")
        print(f"Dataset: allenai/soda (default)")
        print(f"Templates: Default English templates")
        print(f"Samples: {SAMPLE_COUNT}")
        print()

        logger.info("Initializing DatasetBuilder with default settings")
        dataset_builder = DatasetBuilder(
            use_openrouter=True,
            openrouter_api_key=OPENROUTER_API_KEY,
            openrouter_max_workers=2
        )

        # Build CAA dataset with all defaults
        logger.info("Starting CAA dataset generation with default settings")
        try:
            dataset_file = dataset_builder.build_caa_dataset(
                model=DATASET_MODEL,
                personality=PERSONALITY,
                output_dir=DATASET_OUTPUT,
                limit_samples=SAMPLE_COUNT,
                dataset_name="allenai/soda",  # Default dataset
                temperature=0.7,
                max_tokens=100
            )
            print(f"\n[SUCCESS] CAA dataset generated: {dataset_file}")
            logger.info(f"CAA dataset generated successfully: {dataset_file}")

            # Display sample data from generated dataset
            print("\n" + "-"*80)
            print("DATASET SAMPLES (First 2 examples)")
            print("-"*80)
            import json
            with open(dataset_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 2:  # Show only first 2 samples
                        break
                    sample = json.loads(line)
                    print(f"\n[Sample {i+1}]")
                    print(f"Question: {sample['question'][:200]}...")
                    print(f"Positive answer: {sample['positive']}")
                    print(f"Neutral answer: {sample['neutral']}")
            print("-"*80)

        except Exception as e:
            logger.error(f"Failed to generate CAA dataset: {e}")
            raise

    # =========================================================================
    # STEP 2: Extract CAA Steering Vector (Fast Method)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: Extracting Steering Vector")
    print("="*80)
    print(f"Model: {STEERING_MODEL}")
    print(f"Dataset: {dataset_file}")
    print(f"Method: CAA (Mean Contrastive - Fast)")
    print()

    logger.info("Initializing SteeringExtractor")
    extractor = SteeringExtractor()

    # Extract steering vector using CAA method
    logger.info("Starting CAA steering vector extraction")
    try:
        # Use middle layer for gemma-3-270m-it (18 layers total)
        target_layers = ["model.layers.9.mlp.down_proj"]

        extractor.extract_steering_vector(
            model_name=STEERING_MODEL,
            layers=target_layers,
            dataset_path=dataset_file,
            output_path=STEERING_VECTOR_PATH,
            method="mean_contrastive",
            normalize=True
        )
        print(f"\n[SUCCESS] Steering vector extracted: {STEERING_VECTOR_PATH}")
        logger.info(f"Steering vector extracted successfully: {STEERING_VECTOR_PATH}")
    except Exception as e:
        logger.error(f"Failed to extract steering vector: {e}")
        raise

    # =========================================================================
    # STEP 3: Apply Steering and Test
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: Testing Steering")
    print("="*80)
    print(f"Model: {STEERING_MODEL}")
    print(f"Test Input: 'Hello, how are you?'")
    print()

    logger.info("Initializing SteeringApplier")
    applier = SteeringApplier()

    # Test input
    test_input = "Hello, how are you?"

    # Generate response WITHOUT steering
    print("Generating response WITHOUT steering...")
    logger.info("Generating baseline response")
    try:
        loader = LLMLoader()
        model, tokenizer = loader.load_model(STEERING_MODEL)

        # Prepare prompt
        messages = [{"role": "user", "content": test_input}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        response_baseline = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\n[BASELINE] Input: {test_input}")
        print(f"[BASELINE] Response: {response_baseline}")

        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        logger.error(f"Failed to generate baseline response: {e}")
        raise

    # Generate response WITH steering
    print("\nGenerating response WITH steering (Extroversion)...")
    logger.info("Generating steered response")
    try:
        response_steered = applier.apply_steering(
            model_name=STEERING_MODEL,
            steering_vector_path=STEERING_VECTOR_PATH,
            input_text=test_input,
            max_new_tokens=100,
            strength=1.5,
            temperature=0.7
        )
        print(f"\n[STEERED] Input: {test_input}")
        print(f"[STEERED] Response: {response_steered}")
    except Exception as e:
        logger.error(f"Failed to generate steered response: {e}")
        raise

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Dataset: {dataset_file}")
    print(f"Steering Vector: {STEERING_VECTOR_PATH}")
    print(f"\nBaseline Response: {response_baseline}")
    print(f"Steered Response (Extroversion): {response_steered}")
    print("\nSimple workflow completed successfully!")
    print("="*80)

    logger.info("Simple workflow completed successfully")

if __name__ == "__main__":
    main()
