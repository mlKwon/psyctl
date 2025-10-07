"""CAA dataset loader for steering vector extraction."""

import json
from pathlib import Path
from typing import Iterator, List, Tuple

from transformers import AutoTokenizer

from psyctl.core.logger import get_logger


class CAADatasetLoader:
    """
    Load and process CAA (Contrastive Activation Addition) dataset.

    CAA datasets are JSONL files containing contrastive prompt pairs for
    personality steering vector extraction. Each entry has a question and
    two answer options (positive and neutral personality).

    Attributes:
        logger: Logger instance for debugging
    """

    def __init__(self):
        """Initialize CAADatasetLoader with logger."""
        self.logger = get_logger("caa_dataset_loader")

    def load(self, dataset_path: Path) -> List[dict]:
        """
        Load CAA dataset from JSONL file.

        Args:
            dataset_path: Path to dataset directory or JSONL file

        Returns:
            List of dataset entries

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If dataset format is invalid

        Example:
            >>> loader = CAADatasetLoader()
            >>> dataset = loader.load(Path("./dataset/caa"))
        """
        # If path is directory, find JSONL file
        if dataset_path.is_dir():
            jsonl_files = list(dataset_path.glob("*.jsonl"))
            if not jsonl_files:
                raise FileNotFoundError(
                    f"No JSONL files found in directory: {dataset_path}"
                )
            if len(jsonl_files) > 1:
                self.logger.warning(
                    f"Multiple JSONL files found, using first: {jsonl_files[0].name}"
                )
            dataset_file = jsonl_files[0]
        else:
            dataset_file = dataset_path

        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

        self.logger.info(f"Loading dataset from: {dataset_file}")

        dataset = []
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())

                    # Validate required fields
                    required_fields = ['question', 'positive', 'neutral']
                    missing_fields = [
                        field for field in required_fields if field not in entry
                    ]
                    if missing_fields:
                        raise ValueError(
                            f"Missing required fields: {missing_fields}"
                        )

                    dataset.append(entry)

                except json.JSONDecodeError as e:
                    self.logger.error(
                        f"Invalid JSON at line {line_num}: {e}"
                    )
                    raise ValueError(f"Invalid JSON format at line {line_num}") from e
                except ValueError as e:
                    self.logger.error(
                        f"Invalid entry at line {line_num}: {e}"
                    )
                    raise

        self.logger.info(f"Loaded {len(dataset)} entries from dataset")
        return dataset

    def create_prompts(
        self, dataset: List[dict], tokenizer: AutoTokenizer
    ) -> Tuple[List[str], List[str]]:
        """
        Create positive and neutral prompt pairs from dataset.

        Args:
            dataset: List of dataset entries (from load())
            tokenizer: Tokenizer for applying chat template

        Returns:
            Tuple of (positive_prompts, neutral_prompts)

        Example:
            >>> loader = CAADatasetLoader()
            >>> dataset = loader.load(Path("./dataset/caa"))
            >>> pos_prompts, neu_prompts = loader.create_prompts(dataset, tokenizer)
        """
        self.logger.info(f"Creating prompts from {len(dataset)} dataset entries")

        positive_prompts = []
        neutral_prompts = []

        for entry in dataset:
            question = entry['question']
            positive_answer = entry['positive']
            neutral_answer = entry['neutral']

            # Create full prompts by combining question with answers
            positive_prompt = self._build_prompt(
                question, positive_answer, tokenizer
            )
            neutral_prompt = self._build_prompt(
                question, neutral_answer, tokenizer
            )

            positive_prompts.append(positive_prompt)
            neutral_prompts.append(neutral_prompt)

        self.logger.info(
            f"Created {len(positive_prompts)} positive and {len(neutral_prompts)} neutral prompts"
        )
        return positive_prompts, neutral_prompts

    def _build_prompt(
        self, question: str, answer: str, tokenizer: AutoTokenizer
    ) -> str:
        """
        Build a complete prompt from question and answer.

        Args:
            question: Question text from dataset
            answer: Answer option (e.g., "(1" or "(2")
            tokenizer: Tokenizer for chat template

        Returns:
            Complete prompt string ready for inference
        """
        # Combine question with answer
        full_text = f"{question}{answer}"

        # Try to apply chat template if available
        try:
            messages = [{"role": "user", "content": full_text}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            # Fallback to raw text if chat template fails
            self.logger.debug(
                f"Chat template failed, using raw text: {e}"
            )
            prompt = full_text

        return prompt

    def get_batch_iterator(
        self, prompts: List[str], batch_size: int
    ) -> Iterator[List[str]]:
        """
        Create batch iterator for prompts.

        Args:
            prompts: List of prompt strings
            batch_size: Number of prompts per batch

        Yields:
            Batches of prompts

        Example:
            >>> loader = CAADatasetLoader()
            >>> for batch in loader.get_batch_iterator(prompts, batch_size=16):
            ...     # Process batch
            ...     pass
        """
        for i in range(0, len(prompts), batch_size):
            yield prompts[i : i + batch_size]

    def get_dataset_info(self, dataset_path: Path) -> dict:
        """
        Get information about the dataset without loading all data.

        Args:
            dataset_path: Path to dataset directory or JSONL file

        Returns:
            Dictionary with dataset information

        Example:
            >>> loader = CAADatasetLoader()
            >>> info = loader.get_dataset_info(Path("./dataset/caa"))
            >>> print(info['num_samples'], info['file_size_mb'])
        """
        # Find dataset file
        if dataset_path.is_dir():
            jsonl_files = list(dataset_path.glob("*.jsonl"))
            if not jsonl_files:
                raise FileNotFoundError(
                    f"No JSONL files found in directory: {dataset_path}"
                )
            dataset_file = jsonl_files[0]
        else:
            dataset_file = dataset_path

        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

        # Count lines for num_samples
        with open(dataset_file, 'r', encoding='utf-8') as f:
            num_samples = sum(1 for _ in f)

        # Get file size
        file_size_bytes = dataset_file.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)

        info = {
            'file_path': str(dataset_file),
            'num_samples': num_samples,
            'file_size_bytes': file_size_bytes,
            'file_size_mb': round(file_size_mb, 2),
        }

        self.logger.debug(f"Dataset info: {info}")
        return info

    def validate_dataset(self, dataset_path: Path) -> bool:
        """
        Validate dataset format and structure.

        Args:
            dataset_path: Path to dataset directory or JSONL file

        Returns:
            True if dataset is valid, False otherwise

        Example:
            >>> loader = CAADatasetLoader()
            >>> is_valid = loader.validate_dataset(Path("./dataset/caa"))
        """
        try:
            dataset = self.load(dataset_path)

            if len(dataset) == 0:
                self.logger.error("Dataset is empty")
                return False

            # Check first entry structure
            first_entry = dataset[0]
            required_fields = ['question', 'positive', 'neutral']
            for field in required_fields:
                if field not in first_entry:
                    self.logger.error(f"Missing required field: {field}")
                    return False

            self.logger.info("Dataset validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Dataset validation failed: {e}")
            return False
