"""
Dataset Builder for Personality Steering Vector Extraction

This module implements the DatasetBuilder class which creates CAA (Contrastive Activation Addition)
datasets for personality steering vector extraction. The CAA method compares responses from
models with different personality traits to identify steering vectors that can modify model behavior.

Key Concepts:
- CAA (Contrastive Activation Addition): A method to extract steering vectors by comparing
  model responses with different personality traits
- Personality Steering: Modifying LLM behavior to exhibit specific personality characteristics
- Contrastive Learning: Learning from pairs of positive/negative examples

Workflow:
1. Load a base model and tokenizer
2. Load conversation dataset (allenai/soda)
3. Generate personality-specific prompts using P2 class
4. Create contrastive pairs (positive vs neutral personality)
5. Save as JSONL format for training

Example Usage:
    builder = DatasetBuilder()
    builder.build_caa_dataset(
        model="meta-llama/Llama-3.2-3B-Instruct",
        personality="Extroversion",
        output_dir=Path("./dataset"),
        limit_samples=1000
    )

References:
- CAA Paper: https://arxiv.org/abs/2206.07550
- SoDA Dataset: https://huggingface.co/datasets/allenai/soda
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional

from datasets import load_dataset
from tqdm import tqdm, trange

from psyctl.core.logger import get_logger
from psyctl.core.prompt import P2
from psyctl.models.llm_loader import LLMLoader


class DatasetBuilder:
    """
    Build CAA datasets for personality steering vector extraction.

    This class implements the Contrastive Alignment Analysis (CAA) dataset generation
    process. It creates training data by comparing model responses with different
    personality traits, enabling the extraction of steering vectors that can modify
    model behavior to exhibit specific personality characteristics.

    Attributes:
        llm_loader (LLMLoader): Loader for Hugging Face models
        p2 (P2): Personality prompt generator
        logger: Logger instance for debugging and monitoring
        dataset: Loaded conversation dataset (allenai/soda)
        model: Loaded language model
        tokenizer: Model tokenizer
        personality (str): Target personality trait for steering

    Methods:
        build_caa_dataset: Main method to build CAA dataset
        _load_model: Load model and tokenizer
        _load_dataset: Load conversation dataset
        _generate_sample_context: Generate conversation contexts
        _get_answer: Generate personality-specific responses
        _gen_caa_data: Create contrastive data pairs
        _save_sample_to_jsonl: Save data to JSONL file
        _build_caa_dataset: Core dataset building logic
    """

    def __init__(self):
        """
        Initialize DatasetBuilder with required components.

        Initializes the LLM loader, logger, and placeholder attributes for
        model, tokenizer, dataset, and P2 personality generator.
        """
        self.llm_loader = LLMLoader()
        self.p2 = None
        self.logger = get_logger("dataset_builder")
        self.dataset = None
        self.model = None
        self.tokenizer = None

    def build_caa_dataset(
        self, model: str, personality: str, output_dir: Path, limit_samples: int
    ) -> int:
        """
        Build CAA dataset for given personality traits.

        This is the main entry point for CAA dataset generation. It orchestrates
        the entire process from model loading to dataset creation.

        Args:
            model (str): Hugging Face model identifier (e.g., "meta-llama/Llama-3.2-3B-Instruct")
            personality (str): Target personality trait (e.g., "Extroversion", "Introversion")
            output_dir (Path): Directory to save the generated dataset
            limit_samples (int): Maximum number of samples to generate (0 for unlimited)

        Returns:
            int: Number of generated samples

        Raises:
            Exception: If any step in the dataset building process fails

        Example:
            >>> builder = DatasetBuilder()
            >>> num_samples = builder.build_caa_dataset(
            ...     model="meta-llama/Llama-3.2-3B-Instruct",
            ...     personality="Extroversion",
            ...     output_dir=Path("./dataset"),
            ...     limit_samples=1000
            ... )
            >>> print(f"Generated {num_samples} samples")
        """
        self.logger.info(f"Building CAA dataset for model: {model}")
        self.logger.info(f"Personality traits: {personality}")
        self.logger.info(f"Output directory: {output_dir}")

        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created output directory: {output_dir}")
            self.personality = personality

            # 1. Load model
            self._load_model(model)
            self.p2 = P2(self.model, self.tokenizer)

            # 2. Load dataset
            self._load_dataset()

            # 3. Build CAA dataset
            num_samples = self._build_caa_dataset(output_dir, limit_samples)

            self.logger.info(f"Finished building CAA dataset")
            return num_samples

        except Exception as e:
            self.logger.error(f"Failed to build CAA dataset: {e}")
            raise

    def _load_model(self, model_name: str) -> None:
        """
        Load model and tokenizer from Hugging Face.

        Args:
            model_name (str): Hugging Face model identifier

        Raises:
            Exception: If model loading fails
        """
        self.model, self.tokenizer = self.llm_loader.load_model(model_name)
        self.logger.info(f"Loaded model: {model_name}")
        self.logger.info(f"Loaded tokenizer: {model_name}")

    def _load_dataset(self) -> None:
        """
        Load the SoDA conversation dataset.

        Loads the allenai/soda dataset which contains conversational data
        with speakers, dialogue, and narrative context.

        Raises:
            Exception: If dataset loading fails
        """
        # Log HF_TOKEN status for debugging
        import os

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            # Mask the token for security (show first 4 and last 4 characters)
            masked_token = (
                f"{hf_token[:4]}...{hf_token[-4:]}" if len(hf_token) > 8 else "***"
            )
            self.logger.info(f"HF_TOKEN found: {masked_token}")
        else:
            self.logger.warning("HF_TOKEN not found in environment variables")

        try:
            dataset_name = "allenai/soda"
            dataset = load_dataset(dataset_name, split="train")
            self.dataset = dataset
            self.logger.info(f"Loaded dataset: {dataset_name}")
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise

    def _generate_sample_context(
        self, limit_samples: int = 0
    ) -> Generator[Dict[str, str], None, None]:
        """
        Generate conversation contexts from the SoDA dataset.

        Iterates through the dataset to extract valid conversation contexts
        with at least 2 speakers and narrative content. Creates structured
        context data for personality-specific response generation.

        Args:
            limit_samples (int): Maximum number of contexts to generate (0 for unlimited)

        Yields:
            Dict[str, str]: Context dictionary with keys:
                - char_name: Character who will respond
                - user_name: Character who asked the question
                - situation: Combined narrative and dialogue context

        Note:
            Filters out entries with insufficient speakers or missing narrative
        """
        num_generated = 0

        # Calculate total iterations for tqdm
        total = len(self.dataset) if limit_samples == 0 else limit_samples
        pbar = tqdm(range(len(self.dataset)), desc="Generating samples", total=total)
        for idx in pbar:
            data = self.dataset[idx]
            # 화자 최소 2명 보장
            if len(data["speakers"]) < 2 or len(data["dialogue"]) < 1:
                continue
            asker = data["speakers"][0]
            answerer = data["speakers"][1]  # 두 번째 인물
            narrative = data["narrative"] or ""
            if narrative == "":
                continue
            query = data["dialogue"][0]
            situation = f"{narrative}\n{asker}: {query}\n"
            yield {"char_name": answerer, "user_name": asker, "situation": situation}

            num_generated += 1
            # Update progress bar description with actual count
            pbar.set_description(f"Generating samples ({num_generated})")

            if limit_samples > 0 and num_generated >= limit_samples:
                break
        pbar.close()

        self.logger.info(f"Finished generating samples.")

    def _get_answer(
        self,
        user_name: str,
        char_name: str,
        p2: str,
        situation: str,
        verbose: bool = False,
    ) -> str:
        """
        Generate personality-specific response for a given situation.

        Creates a role-playing prompt that instructs the model to respond as a character
        with specific personality traits in a given conversational context.

        Args:
            user_name (str): Name of the user/asker in the conversation
            char_name (str): Name of the character who will respond
            p2 (str): Personality description generated by P2 class
            situation (str): Conversational context and situation
            verbose (bool): Whether to print the generated prompt for debugging

        Returns:
            str: Generated response from the model

        Note:
            The prompt structure follows a role-playing format with clear instructions
            for the model to adopt the character's personality and respond appropriately.
        """
        rp_prompt = [
            "# Overview",
            "This is role playing session.",
            f"Your(Assistant or Model) role is {char_name}. You have to pretend to be {char_name}.",
            f"User's role is {user_name}.",
            f"Write short reaction of {char_name} within the situation in one sentence.",
            "",
            f"# About {char_name}.",
            f"{p2}",
            "",
            "",
            "# Situation",
            f"{situation}",
        ]
        prompt = "\n".join(rp_prompt)
        if verbose:
            print(prompt)

        # Use the same approach as P2._get_result
        messages = [{"role": "user", "content": prompt}]

        # 1. Convert user message to chat template
        try:
            tokenized_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                return_tensors=None,
            )
            tokenized = self.tokenizer(
                tokenized_input, return_tensors="pt", add_special_tokens=False
            )
        except Exception:
            tokenized = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=True
            )

        # Move tensors to the same device as the model
        device = next(self.model.parameters()).device
        tokenized["input_ids"] = tokenized["input_ids"].to(device)
        tokenized["attention_mask"] = tokenized["attention_mask"].to(device)

        # 2. Generate response
        outputs = self.model.generate(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # 3. Decode the generated text
        len_input = tokenized["input_ids"][0].shape[0]
        output_text = self.tokenizer.decode(
            outputs[0, len_input:], skip_special_tokens=True
        )
        return output_text

    def _gen_caa_data(
        self, char_name: str, situation: str, answer_1: str, answer_2: str
    ) -> str:
        """
        Generate contrastive CAA data template.

        Creates a training example from a single situation by presenting
        the model with two different personality-based responses as options.
        This enables contrastive learning for steering vector extraction.

        Args:
            char_name (str): Name of the character in the situation
            situation (str): Conversational context and situation
            answer_1 (str): First personality response option
            answer_2 (str): Second personality response option

        Returns:
            str: Template containing the situation, question, and two answer options.

        Note:
            The responses are cleaned (stripped and newlines removed) before
            being inserted into the template. The template format follows
            a multiple-choice question structure.
        """
        answer_1 = answer_1.strip().replace("\n", "")
        answer_2 = answer_2.strip().replace("\n", "")
        template = "\n".join(
            [
                "[Situation]",
                f"{situation.strip()}",
                "[Question]",
                f"You are {char_name}. What would your response be in this situation?",  # Fixed grammar
                f"1. {answer_1}",
                f"2. {answer_2}",
                "[Answer]",
            ]
        )
        return template

    def _save_sample_to_jsonl(self, sample: Dict[str, str], output_file: Path) -> None:
        """
        Save CAA data samples to JSONL file.

        Appends the generated contrastive data pairs to a JSONL file
        for later use in training steering vector extraction models.

        Args:
            sample (Dict[str, str]): Dictionary containing question, positive, and neutral keys
            output_file (Path): Path to the output JSONL file

        Note:
            Uses UTF-8 encoding and ensures proper JSON formatting with
            non-ASCII character support.
        """
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def _build_caa_dataset(self, output_dir: Path, limit_samples: int) -> int:
        """
        Core CAA dataset building logic.

        This is the main implementation of the CAA dataset generation process.
        It iterates through conversation contexts, generates personality-specific
        responses using P2 prompts, creates contrastive pairs, and saves them
        to a timestamped JSONL file.

        Args:
            output_dir (Path): Directory to save the generated dataset
            limit_samples (int): Maximum number of samples to generate

        Returns:
            int: Number of successfully generated samples

        Note:
            Creates a timestamped output file to avoid overwriting previous datasets.
            Uses P2 class to generate personality-specific character descriptions
            and creates contrastive pairs between target personality and neutral personality.
        """

        self.logger.info(f"Building CAA dataset...")
        self.logger.info(f"Limit samples: {limit_samples}")
        self.logger.info(f"Output directory: {output_dir}")

        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"caa_dataset_{datetime_str}.jsonl"
        self.logger.info(f"Output file: {output_file}")

        num_generated = 0

        # Generate personality-specific character descriptions using P2
        # Positive: Target personality trait (e.g., "Extroversion")
        # Neutral: Baseline personality for contrast
        positive_p2 = self.p2.build("Xylo", self.personality)
        neutral_p2 = self.p2.build("Xylo", "Normal")

        # Create templates with placeholder for character name
        positive_template = positive_p2.replace("Xylo", "{{char}}").replace(
            "Xylo", "{{char}}"
        )
        neutral_template = neutral_p2.replace("Xylo", "{{char}}").replace(
            "Xylo", "{{char}}"
        )

        for context in self._generate_sample_context(limit_samples):
            # Extract context information
            user_name = context["user_name"]
            char_name = context["char_name"]
            situation = context["situation"]

            # Replace placeholder with actual character name
            positive = positive_template.replace("{{char}}", char_name)
            neutral = neutral_template.replace("{{char}}", char_name)

            # Generate responses for both personality types
            answer_positive = self._get_answer(
                user_name, char_name, positive, situation, verbose=False
            )
            answer_neutral = self._get_answer(
                user_name, char_name, neutral, situation, verbose=False
            )

            # Create contrastive training pairs
            sample = {}
            if num_generated % 2 == 0:
                question = self._gen_caa_data(
                    char_name, situation, answer_positive, answer_neutral
                )
                sample["question"] = question
                sample["positive"] = "(1"
                sample["neutral"] = "(2"
            else:
                question = self._gen_caa_data(
                    char_name, situation, answer_neutral, answer_positive
                )
                sample["question"] = question
                sample["positive"] = "(2"
                sample["neutral"] = "(1"

            # Save to JSONL file
            self._save_sample_to_jsonl(sample, output_file)
            num_generated += 1

        self.logger.info(
            f"Finished building CAA dataset. Total samples: {num_generated}"
        )
        return num_generated


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of DatasetBuilder for CAA dataset generation.

    This demonstrates how to use the DatasetBuilder class to create
    personality steering datasets for different personality traits.
    """

    from pathlib import Path

    # Initialize builder
    builder = DatasetBuilder()

    # Example: Build dataset for extroversion personality
    try:
        num_samples = builder.build_caa_dataset(
            model="meta-llama/Llama-3.2-3B-Instruct",
            personality="Extroversion",
            output_dir=Path("./dataset/extroversion"),
            limit_samples=100,
        )
        print(f"Successfully generated {num_samples} samples for Extroversion")

    except Exception as e:
        print(f"Failed to build dataset: {e}")

    # Example: Build dataset for introversion personality
    try:
        num_samples = builder.build_caa_dataset(
            model="meta-llama/Llama-3.2-3B-Instruct",
            personality="Introversion",
            output_dir=Path("./dataset/introversion"),
            limit_samples=100,
        )
        print(f"Successfully generated {num_samples} samples for Introversion")

    except Exception as e:
        print(f"Failed to build dataset: {e}")
