"""CAA dataset loader for steering vector extraction."""

import json
from pathlib import Path
from typing import Iterator, List, Tuple, Union

from datasets import load_dataset
from jinja2 import Environment, FileSystemLoader, Template
from transformers import AutoTokenizer

from psyctl.core.logger import get_logger


class CAADatasetLoader:
    """
    Load and process CAA (Contrastive Activation Addition) dataset.

    Dataset format:
    {
        "situation": "Conversation context...",
        "char_name": "Character name",
        "positive": "Full positive personality answer",
        "neutral": "Full neutral personality answer"
    }

    Attributes:
        logger: Logger instance for debugging
        jinja_env: Jinja2 environment for template loading
    """

    def __init__(self):
        """Initialize CAADatasetLoader with logger and Jinja2 environment."""
        self.logger = get_logger("caa_dataset_loader")

        # Setup Jinja2 environment for template loading
        template_dir = Path(__file__).parent.parent / "templates"
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        self.logger.debug(f"Template directory: {template_dir}")

    def load(self, dataset_path: Union[Path, str]) -> List[dict]:
        """
        Load CAA dataset from JSONL file or HuggingFace dataset.

        Args:
            dataset_path: Path to dataset directory/JSONL file or HuggingFace dataset name

        Returns:
            List of dataset entries

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If dataset format is invalid

        Example:
            >>> loader = CAADatasetLoader()
            >>> dataset = loader.load(Path("./dataset/caa"))
            >>> dataset = loader.load("CaveduckAI/steer-personality-rudeness-ko")
        """
        # Check if it's a HuggingFace dataset name
        if isinstance(dataset_path, str) and '/' in dataset_path:
            self.logger.info(f"Loading HuggingFace dataset: {dataset_path}")
            try:
                hf_dataset = load_dataset(dataset_path, split='train')
                dataset = []
                for item in hf_dataset:
                    # Convert HF dataset format to CAA format
                    entry = {
                        'question': item.get('question', ''),
                        'positive': item.get('positive', ''),
                        'neutral': item.get('neutral', '')
                    }
                    dataset.append(entry)
                self.logger.info(f"Loaded {len(dataset)} entries from HuggingFace dataset")
                return dataset
            except Exception as e:
                raise ValueError(f"Failed to load HuggingFace dataset: {e}")

        # Convert string to Path for local files
        if isinstance(dataset_path, str):
            dataset_path = Path(dataset_path)

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
                    required_fields = ['situation', 'char_name', 'positive', 'neutral']
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
        self, dataset: List[dict], tokenizer: AutoTokenizer, format_type: str = "index"
    ) -> Tuple[List[str], List[str]]:
        """
        Create positive and neutral prompt pairs from dataset.

        Args:
            dataset: List of dataset entries
            tokenizer: Tokenizer for applying chat template
            format_type: Prompt format type
                - "index": Multiple choice with indices (for CAA)
                - "direct": Direct answer (for BiPO full text)

        Returns:
            Tuple of (positive_prompts, neutral_prompts)

        Example:
            >>> loader = CAADatasetLoader()
            >>> dataset = loader.load(Path("./dataset/caa"))
            >>> pos_prompts, neu_prompts = loader.create_prompts(dataset, tokenizer)
        """
        self.logger.info(f"Creating prompts from {len(dataset)} dataset entries (format: {format_type})")

        positive_prompts = []
        neutral_prompts = []

        for entry in dataset:
            situation = entry['situation']
            char_name = entry['char_name']
            positive_answer = entry['positive']
            neutral_answer = entry['neutral']

            # Create prompts based on format type
            if format_type == "index":
                # CAA format: show both answers with indices, append index
                positive_prompt = self._build_prompt_with_choices(
                    situation, char_name, positive_answer, neutral_answer, "(1", tokenizer
                )
                neutral_prompt = self._build_prompt_with_choices(
                    situation, char_name, positive_answer, neutral_answer, "(2", tokenizer
                )
            elif format_type == "direct":
                # BiPO format: direct answer without choices
                positive_prompt = self._build_prompt_direct(
                    situation, char_name, positive_answer, tokenizer
                )
                neutral_prompt = self._build_prompt_direct(
                    situation, char_name, neutral_answer, tokenizer
                )
            else:
                raise ValueError(f"Unknown format_type: {format_type}")

            positive_prompts.append(positive_prompt)
            neutral_prompts.append(neutral_prompt)

        self.logger.info(
            f"Created {len(positive_prompts)} positive and {len(neutral_prompts)} neutral prompts"
        )
        return positive_prompts, neutral_prompts

    def _build_prompt_with_choices(
        self, situation: str, char_name: str, answer_1: str, answer_2: str,
        selected: str, tokenizer: AutoTokenizer
    ) -> str:
        """
        Build CAA-style prompt with multiple choices.

        Args:
            situation: Situation description
            char_name: Character name
            answer_1: First answer option
            answer_2: Second answer option
            selected: Which answer is selected ("(1" or "(2")
            tokenizer: Tokenizer for chat template

        Returns:
            Complete prompt with choices and selection
        """
        # Load template
        template = self.jinja_env.get_template('caa_question.j2')
        question = template.render(
            char_name=char_name,
            situation=situation.strip(),
            answer_1=answer_1.strip().replace("\n", ""),
            answer_2=answer_2.strip().replace("\n", "")
        )

        # Append selected index
        full_text = f"{question}{selected}"

        # Apply chat template if available
        try:
            messages = [{"role": "user", "content": full_text}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            self.logger.debug(f"Chat template failed, using raw text: {e}")
            prompt = full_text

        return prompt

    def _build_prompt_direct(
        self, situation: str, char_name: str, answer: str, tokenizer: AutoTokenizer
    ) -> str:
        """
        Build BiPO-style prompt with direct answer (no choices shown).

        Args:
            situation: Situation description
            char_name: Character name
            answer: The answer text
            tokenizer: Tokenizer for chat template

        Returns:
            Complete prompt with direct answer
        """
        # Simple format for BiPO
        full_text = f"[Situation]\n{situation}\n[Question]\nYou are {char_name}. What would your response be in this situation?\n[Answer]\n{answer}"

        # Apply chat template if available
        try:
            messages = [{"role": "user", "content": full_text}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            self.logger.debug(f"Chat template failed, using raw text: {e}")
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
