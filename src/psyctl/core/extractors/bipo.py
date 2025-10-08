"""Bi-directional Preference Optimization (BiPO) extractor."""

import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer

from psyctl.config import INFERENCE_BATCH_SIZE
from psyctl.core.caa_dataset_loader import CAADatasetLoader
from psyctl.core.extractors.base import BaseVectorExtractor
from psyctl.core.layer_accessor import LayerAccessor
from psyctl.core.logger import get_logger


class BiPOVectorExtractor(BaseVectorExtractor):
    """
    Extract steering vectors using Bi-directional Preference Optimization.

    This extractor implements the BiPO method by optimizing a steering vector
    through gradient descent based on preference learning between positive
    and neutral personality responses.

    Algorithm:
    1. Initialize learnable steering vector v with zeros
    2. For each epoch:
        - Randomly sample direction d ∈ {-1, 1}
        - Compute log probabilities with and without steering
        - Optimize BiPO loss: -log(sigmoid(β * d * (ratio_pos - ratio_neg)))
    3. Return optimized steering vector

    The BiPO loss encourages the model to prefer positive responses when
    steering is applied in the positive direction, and vice versa.

    Attributes:
        dataset_loader: Loader for CAA dataset
        layer_accessor: Accessor for dynamic layer retrieval
        logger: Logger instance
    """

    def __init__(self):
        """Initialize BiPOVectorExtractor."""
        self.dataset_loader = CAADatasetLoader()
        self.layer_accessor = LayerAccessor()
        self.logger = get_logger("bipo_extractor")

    def extract(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        layers: List[str],
        dataset_path: Optional[Union[Path, str]] = None,
        dataset: Optional[List[dict]] = None,
        batch_size: int = None,
        normalize: bool = False,
        lr: float = 5e-4,
        beta: float = 0.1,
        epochs: int = 10,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract steering vectors using BiPO optimization.

        Args:
            model: Loaded language model
            tokenizer: Model tokenizer
            layers: List of layer paths (e.g., ["model.layers[13].mlp"])
            dataset_path: Path to CAA dataset or HuggingFace dataset name (optional if dataset provided)
            dataset: Pre-loaded dataset as list of dicts (optional if dataset_path provided)
            batch_size: Batch size for training (default: from config)
            normalize: Whether to normalize vectors to unit length
            lr: Learning rate for AdamW optimizer (default: 5e-4)
            beta: Temperature parameter for BiPO loss (default: 0.1)
            epochs: Number of training epochs (default: 10)
            **kwargs: Additional parameters (unused)

        Returns:
            Dictionary mapping layer names to steering vectors

        Raises:
            ValueError: If neither dataset_path nor dataset is provided, or both are provided

        Example:
            >>> extractor = BiPOVectorExtractor()
            >>> # Using dataset_path
            >>> vectors = extractor.extract(
            ...     model=model,
            ...     tokenizer=tokenizer,
            ...     layers=["model.layers[13].mlp"],
            ...     dataset_path=Path("./dataset/caa"),
            ...     lr=5e-4,
            ...     beta=0.1,
            ...     epochs=10
            ... )
            >>> # Using pre-loaded dataset
            >>> dataset = [{"question": "...", "positive": "...", "neutral": "..."}]
            >>> vectors = extractor.extract(
            ...     model=model,
            ...     tokenizer=tokenizer,
            ...     layers=["model.layers[13].mlp"],
            ...     dataset=dataset,
            ...     lr=5e-4,
            ...     beta=0.1,
            ...     epochs=10
            ... )
        """
        # Validate dataset parameters
        if dataset is not None and dataset_path is not None:
            raise ValueError("Cannot provide both 'dataset' and 'dataset_path'. Choose one.")
        if dataset is None and dataset_path is None:
            raise ValueError("Must provide either 'dataset' or 'dataset_path'.")

        if batch_size is None:
            batch_size = INFERENCE_BATCH_SIZE

        self.logger.info(f"Extracting BiPO steering vectors from {len(layers)} layers")
        self.logger.info(f"Dataset: {'pre-loaded' if dataset else dataset_path}")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Learning rate: {lr}")
        self.logger.info(f"Beta: {beta}")
        self.logger.info(f"Epochs: {epochs}")
        self.logger.info(f"Normalize: {normalize}")

        # 1. Validate layers
        self.logger.info("Validating layer paths...")
        if not self.layer_accessor.validate_layers(model, layers):
            raise ValueError("Some layer paths are invalid")

        # 2. Load dataset if not provided
        if dataset is None:
            self.logger.info("Loading dataset...")
            dataset = self.dataset_loader.load(dataset_path)
        else:
            self.logger.info("Using pre-loaded dataset...")

        # 3. Freeze model parameters
        for param in model.parameters():
            param.requires_grad = False

        # 4. Extract vectors for each layer
        steering_vectors = {}

        for layer_str in layers:
            self.logger.info(f"Training steering vector for layer: {layer_str}")

            layer_module = self.layer_accessor.get_layer(model, layer_str)
            steering_vec = self._train_steering_vector(
                model=model,
                tokenizer=tokenizer,
                layer_module=layer_module,
                dataset=dataset,
                batch_size=batch_size,
                lr=lr,
                beta=beta,
                epochs=epochs,
            )

            if normalize:
                norm = steering_vec.norm()
                if norm > 1e-8:
                    steering_vec = steering_vec / norm
                    self.logger.debug(f"Normalized vector for '{layer_str}'")
                else:
                    self.logger.warning(
                        f"Vector for '{layer_str}' has near-zero norm, skipping normalization"
                    )

            steering_vectors[layer_str] = steering_vec

            self.logger.info(
                f"Extracted steering vector for '{layer_str}': "
                f"shape={steering_vec.shape}, norm={steering_vec.norm():.4f}"
            )

        self.logger.info(
            f"Successfully extracted {len(steering_vectors)} steering vectors"
        )
        return steering_vectors

    def _train_steering_vector(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        layer_module: nn.Module,
        dataset: List[Dict],
        batch_size: int,
        lr: float,
        beta: float,
        epochs: int,
    ) -> torch.Tensor:
        """
        Train a single steering vector using BiPO optimization.

        Args:
            model: Language model
            tokenizer: Tokenizer
            layer_module: Target layer module
            dataset: Dataset samples
            batch_size: Training batch size
            lr: Learning rate
            beta: BiPO temperature parameter
            epochs: Number of training epochs

        Returns:
            Optimized steering vector
        """
        # Initialize steering vector
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        hidden_size = model.config.hidden_size

        # Log GPU information
        if device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(device)
            gpu_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            self.logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.2f} GB)")
            self.logger.info(f"CUDA device: {device}")
        else:
            self.logger.warning(f"Using CPU: {device} (GPU acceleration not available)")

        v = torch.zeros(hidden_size, requires_grad=True, device=device, dtype=dtype)
        optimizer = AdamW([v], lr=lr, weight_decay=0.05)

        # Prepare dataset
        dataset_samples = self._prepare_dataset(dataset, tokenizer)

        # Training loop
        for epoch in range(epochs):
            random.shuffle(dataset_samples)
            epoch_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(
                range(0, len(dataset_samples), batch_size),
                desc=f"Epoch {epoch + 1}/{epochs}",
            )

            for i in progress_bar:
                batch = dataset_samples[i : i + batch_size]

                optimizer.zero_grad()
                loss = self._compute_bipo_loss(
                    model=model,
                    tokenizer=tokenizer,
                    layer_module=layer_module,
                    batch=batch,
                    v=v,
                    beta=beta,
                )

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "v_norm": f"{v.norm().item():.4f}",
                    }
                )

            avg_loss = epoch_loss / num_batches

            # Log GPU memory usage if using CUDA
            gpu_mem_info = ""
            if device.type == 'cuda':
                allocated = torch.cuda.memory_allocated(device) / (1024**3)
                reserved = torch.cuda.memory_reserved(device) / (1024**3)
                gpu_mem_info = f", GPU_mem: {allocated:.2f}/{reserved:.2f}GB"

            self.logger.info(
                f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Vector_norm={v.norm().item():.4f}{gpu_mem_info}"
            )

        return v.detach().clone()

    def _prepare_dataset(
        self, dataset: List[Dict], tokenizer: AutoTokenizer
    ) -> List[tuple]:
        """
        Prepare dataset samples for BiPO training.

        Args:
            dataset: Raw dataset
            tokenizer: Tokenizer

        Returns:
            List of (question, positive_response, neutral_response) tuples
        """
        samples = []

        for item in dataset:
            question = item["question"]
            positive = item["positive"]
            neutral = item["neutral"]

            samples.append((question, positive, neutral))

        self.logger.info(f"Prepared {len(samples)} training samples")
        return samples

    def _compute_bipo_loss(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        layer_module: nn.Module,
        batch: List[tuple],
        v: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        """
        Compute BiPO loss for a batch.

        Args:
            model: Language model
            tokenizer: Tokenizer
            layer_module: Target layer
            batch: List of (question, positive, neutral) tuples
            v: Current steering vector
            beta: Temperature parameter

        Returns:
            BiPO loss
        """
        # Random direction
        d = random.choice([-1, 1])
        total_loss = 0.0

        for question, positive_resp, negative_resp in batch:
            # Original log probabilities
            log_prob_pos_orig = self._get_response_logprob(
                model, tokenizer, question, positive_resp, layer_module, None
            )
            log_prob_neg_orig = self._get_response_logprob(
                model, tokenizer, question, negative_resp, layer_module, None
            )

            # Steered log probabilities
            log_prob_pos_steered = self._get_response_logprob(
                model, tokenizer, question, positive_resp, layer_module, d * v
            )
            log_prob_neg_steered = self._get_response_logprob(
                model, tokenizer, question, negative_resp, layer_module, d * v
            )

            # BiPO objective
            ratio_pos = log_prob_pos_steered - log_prob_pos_orig
            ratio_neg = log_prob_neg_steered - log_prob_neg_orig

            logits = d * beta * (ratio_pos - ratio_neg)
            loss = -torch.log(torch.sigmoid(logits))
            total_loss += loss

        return total_loss / len(batch)

    def _get_response_logprob(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        question: str,
        response: str,
        layer_module: nn.Module,
        steering: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Calculate log probability of response tokens.

        Args:
            model: Language model
            tokenizer: Tokenizer
            question: Question text
            response: Response text
            layer_module: Target layer
            steering: Steering vector to apply (None for no steering)

        Returns:
            Sum of log probabilities for response tokens
        """
        # Format question with chat template if available
        question_text = self._format_with_chat_template(tokenizer, question)
        full_text = question_text + response

        tokens = tokenizer(
            full_text, return_tensors="pt", max_length=512, truncation=True
        )

        # Calculate question length to identify response tokens
        question_tokens = tokenizer(
            question_text, max_length=512, truncation=True
        )
        question_len = len(question_tokens.input_ids)

        # Register steering hook if needed
        hook_handle = None
        if steering is not None:

            def steering_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                steered_states = hidden_states + steering.unsqueeze(0).unsqueeze(0).to(
                    hidden_states.dtype
                )

                if isinstance(output, tuple):
                    return (steered_states,) + output[1:]
                else:
                    return steered_states

            hook_handle = layer_module.register_forward_hook(steering_hook)

        try:
            # Forward pass
            device = next(model.parameters()).device
            input_ids = tokens.input_ids.to(device)

            if steering is not None:
                logits = model(input_ids).logits
            else:
                with torch.no_grad():
                    logits = model(input_ids).logits

            # Calculate log probabilities for response tokens
            log_probs = F.log_softmax(logits, dim=-1)
            total_logprob = 0.0

            for i in range(question_len - 1, input_ids.size(1) - 1):
                next_token = input_ids[0, i + 1]
                total_logprob += log_probs[0, i, next_token]

            return total_logprob

        finally:
            if hook_handle:
                hook_handle.remove()

    def _format_with_chat_template(
        self, tokenizer: AutoTokenizer, text: str
    ) -> str:
        """
        Format text using model's chat template if available.

        Args:
            tokenizer: Model tokenizer
            text: Text to format

        Returns:
            Formatted text (with chat template if available, otherwise original)
        """
        # Try to use chat template if available
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            try:
                messages = [{"role": "user", "content": text}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                self.logger.debug("Applied chat template for prompt formatting")
                return formatted
            except Exception as e:
                self.logger.warning(
                    f"Failed to apply chat template: {e}. Using raw text."
                )
                return text
        else:
            # Fallback: return raw text
            self.logger.debug("No chat template available, using raw text")
            return text
