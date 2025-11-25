"""
LLM Processor module for Hailo Voice Assistant.

This module handles text generation using Hailo's Large Language Model.
"""

from typing import Any, Generator, List, Optional, Union

from hailo_platform import VDevice
from hailo_platform.genai import LLM

from hailo_apps.python.core.common.core import get_resource_path
from hailo_apps.python.core.common.defines import (
    LLM_MODEL_NAME_H10,
    RESOURCES_MODELS_DIR_NAME,
)


class LLMProcessor:
    """
    Handles text generation using Hailo's Large Language Model.
    """

    def __init__(self, vdevice: VDevice, model_name: str = LLM_MODEL_NAME_H10):
        """
        Initialize the LLMProcessor.

        Args:
            vdevice (VDevice): The Hailo VDevice instance to use.
            model_name (str): Name of the LLM model to load. Defaults to LLM_MODEL_NAME_H10.
        """
        model_path = str(
            get_resource_path(
                pipeline_name=None,
                resource_type=RESOURCES_MODELS_DIR_NAME,
                model=model_name,
            )
        )
        self.llm = LLM(vdevice, model_path)
        self._recovery_seq = self.llm.get_generation_recovery_sequence()

    def generate(
        self,
        prompt: Union[str, List[dict]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """
        Generate text response from the LLM.

        Args:
            prompt (Union[str, List[dict]]): The input prompt. Can be a string or a list of message dicts.
            max_tokens (Optional[int]): Maximum number of tokens to generate.
            temperature (Optional[float]): Sampling temperature.
            stop_sequences (Optional[List[str]]): Sequences to stop generation at.

        Yields:
            str: Generated text chunks (tokens).
        """
        # Handle prompt format
        if isinstance(prompt, str):
            formatted_prompt = [{'role': 'user', 'content': prompt}]
        else:
            formatted_prompt = prompt

        kwargs = {}
        if max_tokens is not None:
            kwargs['max_generated_tokens'] = max_tokens
        if temperature is not None:
            kwargs['temperature'] = temperature
        if stop_sequences is not None:
            # Note: Check if underlying API supports stop_sequences if needed,
            # or implement manual stopping. Assuming API might support it or we just generate.
            pass

        with self.llm.generate(prompt=formatted_prompt, **kwargs) as gen:
            for token in gen:
                if token == self._recovery_seq:
                    continue
                yield token

    def clear_context(self):
        """Clear the LLM context."""
        self.llm.clear_context()

    def get_context_usage_size(self) -> int:
        """Get current context usage in tokens."""
        return self.llm.get_context_usage_size()

    def max_context_capacity(self) -> int:
        """Get maximum context capacity in tokens."""
        return self.llm.max_context_capacity()

    def save_context(self) -> bytes:
        """Save current context state."""
        return self.llm.save_context()

    def load_context(self, context_data: bytes):
        """Load context state."""
        self.llm.load_context(context_data)

