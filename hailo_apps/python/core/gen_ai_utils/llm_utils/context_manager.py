"""
Context management utilities for LLM interactions.

Handles checking context usage, trimming context, and caching context state.
"""

import logging
from pathlib import Path
from typing import Optional

from hailo_platform.genai import LLM

from hailo_apps.python.core.gen_ai_utils.llm_utils.message_formatter import messages_system

# Setup logger
logger = logging.getLogger(__name__)


def check_and_trim_context(
    llm: LLM, context_threshold: float = 0.80, logger_instance: Optional[logging.Logger] = None
) -> bool:
    """
    Check if context needs trimming and clear/reset if needed.

    Uses actual token usage from the LLM API to determine when to clear context.

    Args:
        llm (LLM): The LLM instance to check.
        context_threshold (float): Threshold percentage (0.0-1.0) to trigger clear.
        logger_instance (logging.Logger): Logger to use. Defaults to module logger.

    Returns:
        bool: True if context was cleared, False otherwise.
    """
    log = logger_instance or logger
    try:
        max_capacity = llm.max_context_capacity()
        current_usage = llm.get_context_usage_size()

        # Clear when we reach threshold to leave room for next response
        threshold = int(max_capacity * context_threshold)

        if current_usage < threshold:
            return False

        log.info(
            f"Context at {current_usage}/{max_capacity} tokens ({current_usage*100//max_capacity}%); clearing..."
        )
        llm.clear_context()
        log.info("Context cleared successfully.")
        return True

    except Exception as e:
        log.warning(f"Failed to check/clear context: {e}")
        return False


def print_context_usage(
    llm: LLM, show_always: bool = False, logger_instance: Optional[logging.Logger] = None
) -> None:
    """
    Display context usage statistics.

    Args:
        llm (LLM): The LLM instance.
        show_always (bool): If True, print to user. If False, only log at DEBUG level.
        logger_instance (logging.Logger): Logger to use. Defaults to module logger.
    """
    log = logger_instance or logger
    try:
        max_capacity = llm.max_context_capacity()
        current_usage = llm.get_context_usage_size()
        percentage = (current_usage * 100) // max_capacity if max_capacity > 0 else 0

        # Create visual progress bar
        bar_length = 30
        filled = (current_usage * bar_length) // max_capacity if max_capacity > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)

        usage_str = f"Context: [{bar}] {current_usage}/{max_capacity} tokens ({percentage}%)"

        if show_always:
            print(f"[Info] {usage_str}")
        else:
            log.debug(usage_str)

    except Exception as e:
        log.debug(f"Could not get context usage: {e}")


def get_context_cache_path(tool_name: str, cache_dir: Path) -> Path:
    """
    Get the path to the context cache file for a given tool.

    Args:
        tool_name (str): Name of the tool.
        cache_dir (Path): Directory to store cache files.

    Returns:
        Path: Path to the context cache file.
    """
    cache_filename = f"context_{tool_name}.cache"
    return cache_dir / cache_filename


def save_context_to_cache(
    llm: LLM, tool_name: str, cache_dir: Path, logger_instance: Optional[logging.Logger] = None
) -> bool:
    """
    Save LLM context to a cache file for faster future loading.

    Args:
        llm (LLM): The LLM instance with context to save.
        tool_name (str): Name of the tool.
        cache_dir (Path): Directory to store cache files.
        logger_instance (logging.Logger): Logger to use.

    Returns:
        bool: True if context was saved successfully, False otherwise.
    """
    log = logger_instance or logger
    try:
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)

        cache_path = get_context_cache_path(tool_name, cache_dir)
        log.debug(f"Saving context to cache file: {cache_path}")

        # Get context data from LLM
        context_data = llm.save_context()

        # Save context data to file (binary format)
        with open(cache_path, 'wb') as f:
            f.write(context_data)

        log.info(f"Context cache saved successfully for tool '{tool_name}'")
        return True
    except Exception as e:
        log.warning(f"Failed to save context cache for tool '{tool_name}': {e}")
        return False


def load_context_from_cache(
    llm: LLM, tool_name: str, cache_dir: Path, logger_instance: Optional[logging.Logger] = None
) -> bool:
    """
    Load LLM context from a cache file if it exists.

    Args:
        llm (LLM): The LLM instance to load context into.
        tool_name (str): Name of the tool.
        cache_dir (Path): Directory to read cache files from.
        logger_instance (logging.Logger): Logger to use.

    Returns:
        bool: True if context was loaded successfully, False if file doesn't exist or load failed.
    """
    log = logger_instance or logger
    try:
        cache_path = get_context_cache_path(tool_name, cache_dir)

        if not cache_path.exists():
            log.info(f"No context cache found for tool '{tool_name}' at {cache_path}")
            return False

        log.debug(f"Loading context from cache file: {cache_path}")

        # Read context data from file (binary format)
        with open(cache_path, 'rb') as f:
            context_data = f.read()

        # Load context data into LLM
        llm.load_context(context_data)

        log.info(f"Context cache loaded successfully for tool '{tool_name}'")
        return True
    except Exception as e:
        log.warning(f"Failed to load context cache for tool '{tool_name}': {e}")
        return False


def initialize_system_prompt_context(
    llm: LLM, system_text: str, logger_instance: Optional[logging.Logger] = None
) -> None:
    """
    Initialize LLM context with system prompt by generating a minimal response.

    This adds the system prompt to the LLM's context by sending it and generating
    a single token. We instruct the model to respond with only the end token to
    avoid adding unnecessary content to the context.

    Args:
        llm (LLM): The LLM instance to initialize.
        system_text (str): The system prompt text to add to context.
        logger_instance (logging.Logger): Logger to use.
    """
    log = logger_instance or logger
    try:
        log.info("Initializing system prompt in context...")

        # Build prompt with system message and a request for minimal response
        prompt = [
            messages_system(system_text + " Respond with only a single space character."),
        ]

        # Generate a single token to add the system prompt to context
        # We discard the output since we only need it in context
        generated_tokens = []
        for token in llm.generate(prompt=prompt, max_generated_tokens=1):
            generated_tokens.append(token)

        # Log what was generated for debugging
        generated_text = "".join(generated_tokens)
        log.debug(f"System prompt initialization generated token: {repr(generated_text)}")
        log.info("System prompt successfully added to context")

    except Exception as e:
        log.warning(f"Failed to initialize system prompt context: {e}")
        # Don't raise - allow the application to continue
        # The system prompt will be added normally on first user message

