"""
Dataset Upload to HuggingFace Hub Example

This example demonstrates how to upload a generated CAA dataset to HuggingFace Hub.
The uploaded dataset can be used directly for steering vector extraction without
needing to regenerate it.

Features:
- Automatic dataset card generation with PSYCTL branding
- Metadata tracking (personality, model, sample count, source dataset)
- Public or private repository options
- License specification support

Requirements:
- .env file with HF_TOKEN (with write permission)
- Previously generated CAA dataset file (JSONL format)

Use Cases:
- Share datasets with the community
- Reuse datasets across different projects
- Preserve datasets for reproducibility
- Collaborate with team members
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from psyctl.core.dataset_builder import DatasetBuilder
from psyctl.core.logger import get_logger
from psyctl.core.utils import validate_hf_token

# Initialize logger
logger = get_logger("dataset_upload_example")

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN not found in .env file. Please set it with write permission."
    )

# Configuration
DATASET_FILE = Path(
    "./results/dataset_local/caa_dataset_20250107_143022.jsonl"
)  # Change to your dataset file
REPO_ID = "username/extroversion-caa"  # Change to your HuggingFace username/repo-name
PRIVATE = False  # Set to True for private repository
LICENSE = "mit"  # Options: 'mit', 'apache-2.0', 'cc-by-4.0', 'cc-by-nc-4.0', etc.


def main():
    """Upload CAA dataset to HuggingFace Hub."""

    print("=" * 80)
    print("Dataset Upload to HuggingFace Hub")
    print("=" * 80)
    print(f"Dataset file: {DATASET_FILE}")
    print(f"Repository: {REPO_ID}")
    print(f"Privacy: {'Private' if PRIVATE else 'Public'}")
    print(f"License: {LICENSE}")
    print()

    # =========================================================================
    # Step 1: Validate HuggingFace Token
    # =========================================================================
    print("Step 1: Validating HuggingFace token...")
    try:
        token = validate_hf_token()
        print("[OK] HuggingFace token is valid")
        logger.info("HuggingFace token validated successfully")
    except Exception as e:
        print(f"[ERROR] Token validation failed: {e}")
        print("\nPlease ensure:")
        print("1. HF_TOKEN is set in .env file")
        print("2. Token has 'write' permission")
        print("3. Token is valid (not expired)")
        raise

    # =========================================================================
    # Step 2: Check Dataset File
    # =========================================================================
    print("\nStep 2: Checking dataset file...")
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATASET_FILE}\n"
            f"Please generate a dataset first using:\n"
            f"  python examples/dataset_generation_local.py\n"
            f"or\n"
            f"  python examples/dataset_generation_openrouter.py"
        )

    # Count samples in dataset
    with Path(DATASET_FILE).open(encoding="utf-8") as f:
        sample_count = sum(1 for _ in f)
    file_size = DATASET_FILE.stat().st_size / 1024  # KB

    print("[OK] Dataset file found")
    print(f"     Samples: {sample_count}")
    print(f"     Size: {file_size:.2f} KB")

    # =========================================================================
    # Step 3: Upload to HuggingFace Hub
    # =========================================================================
    print("\nStep 3: Uploading to HuggingFace Hub...")
    print("This may take a few moments...")

    try:
        # Initialize DatasetBuilder
        builder = DatasetBuilder()

        # Upload dataset
        repo_url = builder.upload_to_hub(
            jsonl_file=DATASET_FILE,
            repo_id=REPO_ID,
            private=PRIVATE,
            license=LICENSE,
            token=token,
        )

        print("\n[SUCCESS] Dataset uploaded successfully!")
        print(f"Repository URL: {repo_url}")
        logger.info(f"Dataset uploaded to: {repo_url}")

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        print(f"\n[ERROR] Upload failed: {e}")
        print("\nTroubleshooting:")
        print("- Check if repository name is available")
        print("- Verify HF_TOKEN has write permission")
        print("- Ensure repository ID format: username/repo-name")
        raise

    # =========================================================================
    # Step 4: Usage Instructions
    # =========================================================================
    print("\n" + "=" * 80)
    print("UPLOAD SUMMARY")
    print("=" * 80)
    print(f"Dataset: {DATASET_FILE.name}")
    print(f"Repository: {REPO_ID}")
    print(f"URL: {repo_url}")
    print(f"Samples: {sample_count}")
    print(f"Privacy: {'Private' if PRIVATE else 'Public'}")
    print(f"License: {LICENSE}")
    print()
    print("The dataset card includes:")
    print("- PSYCTL branding and logo")
    print("- Metadata (personality, model, sample count, source dataset)")
    print("- Usage instructions for steering vector extraction")
    print("- References to CAA and P2 papers")
    print()
    print("Next steps:")
    print(f"1. View your dataset at: {repo_url}")
    print("2. Use it for extraction:")
    print("   psyctl extract.steering \\")
    print('     --model "google/gemma-3-270m-it" \\')
    print('     --layer "model.layers.9.mlp.down_proj" \\')
    print(f'     --dataset "{REPO_ID}" \\')
    print('     --output "./steering_vector/out.safetensors"')
    print("=" * 80)


if __name__ == "__main__":
    main()
