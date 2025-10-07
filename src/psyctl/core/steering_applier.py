"""Steering vector applier for text generation."""

from pathlib import Path
from typing import Dict, Optional

import torch

from psyctl.core.layer_accessor import LayerAccessor
from psyctl.core.logger import get_logger
from psyctl.models.llm_loader import LLMLoader
from psyctl.models.vector_store import VectorStore


class SteeringApplier:
    """Apply steering vectors to models for text generation."""

    def __init__(self):
        self.logger = get_logger("steering_applier")
        self.llm_loader = LLMLoader()
        self.vector_store = VectorStore()
        self.layer_accessor = LayerAccessor()

    def apply_steering(
        self,
        model_name: str,
        steering_vector_path: Path,
        input_text: str,
        strength: float = 1.0,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        orthogonal: bool = False,
    ) -> str:
        """
        Apply steering vector and generate text.

        Args:
            model_name: Hugging Face model identifier
            steering_vector_path: Path to steering vector file
            input_text: Input text for generation
            strength: Steering strength multiplier (default: 1.0)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 for greedy)
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            orthogonal: Use orthogonalized addition method

        Returns:
            Generated text string

        Example:
            >>> applier = SteeringApplier()
            >>> result = applier.apply_steering(
            ...     model_name="google/gemma-3-270m-it",
            ...     steering_vector_path=Path("./vector.safetensors"),
            ...     input_text="hello world",
            ...     strength=1.5
            ... )
        """
        self.logger.info(f"Applying steering vector for model: {model_name}")
        self.logger.info(f"Steering vector path: {steering_vector_path}")
        self.logger.info(f"Input text: {input_text}")
        self.logger.info(f"Strength: {strength}, Temperature: {temperature}")

        try:
            # Validate inputs
            if not steering_vector_path.exists():
                raise FileNotFoundError(
                    f"Steering vector file does not exist: {steering_vector_path}"
                )

            # 1. Load model and tokenizer
            self.logger.info("Loading model and tokenizer...")
            model, tokenizer = self.llm_loader.load_model(model_name)

            # 2. Load steering vectors and metadata
            self.logger.info("Loading steering vectors...")
            vectors, metadata = self.vector_store.load_multi_layer(
                steering_vector_path
            )

            # 3. Prepare prompt with chat template
            prompt = self._prepare_prompt(input_text, tokenizer)
            self.logger.debug(f"Prepared prompt: {prompt[:100]}...")

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            prompt_length = inputs.input_ids.shape[1]
            self.logger.debug(f"Prompt length: {prompt_length} tokens")

            # 4. Register hooks for each layer
            self.logger.info(f"Registering hooks for {len(vectors)} layers...")
            hooks = []
            try:
                for layer_name, steer_vec in vectors.items():
                    layer_module = self._get_layer_module(
                        model, layer_name, metadata
                    )
                    hook = self._make_steering_hook(
                        prompt_length, steer_vec, strength, orthogonal
                    )
                    handle = layer_module.register_forward_hook(hook)
                    hooks.append(handle)
                    self.logger.debug(f"Registered hook on: {layer_name}")

                # 5. Generate with steering
                self.logger.info("Generating text with steering...")
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True if temperature > 0 else False,
                        temperature=temperature if temperature > 0 else None,
                        top_p=top_p,
                        top_k=top_k,
                        pad_token_id=tokenizer.pad_token_id
                        or tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=False,  # Required when using hooks
                    )

                # 6. Decode and return
                result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                generated_text = result.replace(prompt, "").strip()

                self.logger.success("Text generation completed successfully")
                return generated_text

            finally:
                # Always remove hooks
                for handle in hooks:
                    handle.remove()
                self.logger.debug("Removed all hooks")

        except Exception as e:
            self.logger.error(f"Failed to apply steering vector: {e}")
            raise

    def _prepare_prompt(self, input_text: str, tokenizer) -> str:
        """
        Prepare prompt with chat template if available.

        Args:
            input_text: User input text
            tokenizer: HuggingFace tokenizer

        Returns:
            Formatted prompt string
        """
        # Try to use chat template if available
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            try:
                messages = [{"role": "user", "content": input_text}]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                self.logger.debug("Applied chat template")
                return prompt
            except Exception as e:
                self.logger.debug(f"Chat template failed, using raw text: {e}")

        # Fallback to raw input
        return input_text

    def _get_layer_module(self, model, layer_name: str, metadata: Dict):
        """
        Get layer module from model using layer name.

        Args:
            model: PyTorch model
            layer_name: Layer path string
            metadata: Metadata from vector file

        Returns:
            PyTorch module
        """
        try:
            return self.layer_accessor.get_layer(model, layer_name)
        except Exception as e:
            self.logger.error(f"Failed to access layer '{layer_name}': {e}")
            raise

    def _make_steering_hook(
        self,
        prompt_length: int,
        steer_vec: torch.Tensor,
        strength: float,
        orthogonal: bool,
    ):
        """
        Create steering hook function (CAA method).

        This implements the CAA method from the PoC code (CAA.ipynb cell-35).
        Setting prompt_length=0 applies steering to all tokens (BiPO-style).

        Args:
            prompt_length: Length of prompt in tokens (steering applied after this)
            steer_vec: Steering vector tensor
            strength: Multiplication strength
            orthogonal: Use orthogonalized addition method

        Returns:
            Hook function for register_forward_hook()
        """

        def hook(module, input, output):
            # Handle tuple output (some layers return (hidden_states, *extra))
            if isinstance(output, tuple):
                out = output[0]
                extra_outputs = output[1:]
            else:
                out = output
                extra_outputs = ()

            # Clone and ensure floating point
            if not torch.is_floating_point(out):
                out = out.float()
            out = out.clone()

            # Prepare steering vector
            steer = steer_vec.to(device=out.device, dtype=out.dtype)
            steer_reshaped = steer.view(1, 1, -1)  # [1, 1, H] for broadcasting

            # Apply steering to tokens after prompt
            if orthogonal:
                # Orthogonalized addition: remove existing component then add steering
                norm_steer = steer / (steer.norm(p=2) + 1e-8)
                proj_coeff = (out[:, prompt_length:, :] * norm_steer).sum(
                    dim=-1, keepdim=True
                )
                proj = proj_coeff * norm_steer
                out[:, prompt_length:, :] = (
                    out[:, prompt_length:, :] - proj
                ) + strength * steer_reshaped
            else:
                # Simple addition
                out[:, prompt_length:, :] = (
                    out[:, prompt_length:, :] + strength * steer_reshaped
                )

            # Return in original format
            if extra_outputs:
                return (out,) + extra_outputs
            else:
                return out

        return hook
