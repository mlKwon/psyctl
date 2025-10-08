"""
PSYCTL End-to-End Example: BiPO Steering with Uploaded Dataset

This script demonstrates using a pre-uploaded HuggingFace dataset for BiPO steering:
1. Load uploaded dataset: CaveduckAI/steer-personality-rudeness-ko
2. Extract BiPO steering vector from a local model
3. Apply steering to test the personality transformation

Requirements:
- .env file with HF_TOKEN
- ~2GB disk space for model cache
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Import PSYCTL components
from psyctl.core.steering_extractor import SteeringExtractor
from psyctl.core.steering_applier import SteeringApplier
from psyctl.core.logger import get_logger

# Initialize logger
logger = get_logger("bipo_uploaded_dataset_example")

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Validate API key
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

# Configuration
STEERING_MODEL = "google/gemma-3-270m-it"  # Local model for steering extraction and application
UPLOADED_DATASET = "CaveduckAI/steer-personality-rudeness-ko"  # Pre-uploaded HuggingFace dataset
RESULTS_DIR = Path("./results")
STEERING_VECTOR_PATH = RESULTS_DIR / "rudeness_bipo_steering.safetensors"

def main():
    """Execute the BiPO workflow with uploaded dataset."""
    parser = argparse.ArgumentParser(description="PSYCTL BiPO Example with Uploaded Dataset")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs for BiPO (default: 10)")
    parser.add_argument("--strength", type=float, default=1.5,
                        help="Steering strength multiplier (default: 1.5)")
    args = parser.parse_args()

    print("="*80)
    print("PSYCTL BiPO Example: Using Uploaded Dataset")
    print("="*80)
    print()

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STEP 1: Extract BiPO Steering Vector from Uploaded Dataset
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: Extracting BiPO Steering Vector")
    print("="*80)
    print(f"Model: {STEERING_MODEL}")
    print(f"Dataset: {UPLOADED_DATASET}")
    print(f"Method: BiPO (Bi-Directional Preference Optimization)")
    print(f"Epochs: {args.epochs}")
    print()

    logger.info("Initializing SteeringExtractor")
    extractor = SteeringExtractor()

    # Extract steering vector using BiPO method
    logger.info("Starting BiPO steering vector extraction")
    try:
        # Determine target layers - use middle layer for gemma-3-270m-it
        target_layers = ["model.layers.9.mlp.down_proj"]  # Middle layer for gemma-3-270m-it (18 layers)

        extractor.extract_steering_vector(
            model_name=STEERING_MODEL,
            layers=target_layers,
            dataset_path=UPLOADED_DATASET,  # Use uploaded HF dataset
            output_path=STEERING_VECTOR_PATH,
            method="bipo",
            normalize=True,
            epochs=args.epochs
        )
        print(f"\n[SUCCESS] Steering vector extracted: {STEERING_VECTOR_PATH}")
        logger.info(f"Steering vector extracted successfully: {STEERING_VECTOR_PATH}")
    except Exception as e:
        logger.error(f"Failed to extract steering vector: {e}")
        raise

    # =========================================================================
    # STEP 2: Apply Steering and Test
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: Applying Steering and Testing")
    print("="*80)
    print(f"Model: {STEERING_MODEL}")
    print(f"Steering Vector: {STEERING_VECTOR_PATH}")
    print(f"Steering Strength: {args.strength}")
    print()

    logger.info("Initializing SteeringApplier")
    applier = SteeringApplier()

    # Test inputs
    test_inputs = [
        "안녕하세요",
        "오늘 날씨가 좋네요",
        "도움이 필요하신가요?"
    ]

    for test_input in test_inputs:
        print(f"\n{'='*80}")
        print(f"Test Input: {test_input}")
        print(f"{'='*80}")

        # Generate response WITHOUT steering (using a simple baseline generation)
        print("\nGenerating response WITHOUT steering...")
        logger.info(f"Generating baseline response for: {test_input}")
        try:
            import torch
            from psyctl.models.llm_loader import LLMLoader
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
            print(f"[BASELINE] {response_baseline}")

            # Clean up
            del model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            logger.error(f"Failed to generate baseline response: {e}")
            raise

        # Generate response WITH steering
        print("\nGenerating response WITH steering...")
        logger.info(f"Generating steered response for: {test_input}")
        try:
            response_steered = applier.apply_steering(
                model_name=STEERING_MODEL,
                steering_vector_path=STEERING_VECTOR_PATH,
                input_text=test_input,
                max_new_tokens=100,
                strength=args.strength,
                temperature=0.7
            )
            print(f"[STEERED] {response_steered}")
        except Exception as e:
            logger.error(f"Failed to generate steered response: {e}")
            raise

        print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"[SUCCESS] Dataset: {UPLOADED_DATASET}")
    print(f"[SUCCESS] Steering Vector: {STEERING_VECTOR_PATH}")
    print(f"[SUCCESS] Training Epochs: {args.epochs}")
    print(f"[SUCCESS] Steering Strength: {args.strength}")
    print("\nBiPO workflow with uploaded dataset completed successfully!")
    print("="*80)

    logger.info("BiPO workflow completed successfully")

if __name__ == "__main__":
    main()
