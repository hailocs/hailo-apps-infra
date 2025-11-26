"""
Context management utilities for LLM interactions.

Handles checking context usage, trimming context, and caching context state.
Provides robust file operations with atomic writes.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

from hailo_platform.genai import LLM

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

        log.debug(
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
    llm: LLM,
    tool_name: str,
    cache_dir: Path,
    logger_instance: Optional[logging.Logger] = None
) -> bool:
    """
    Save LLM context to a cache file for faster future loading.

    Uses atomic writes (write to temp then rename) to prevent corruption.

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
        try:
            context_data = llm.save_context()
        except Exception as e:
             log.warning(f"LLM save_context failed: {e}")
             return False

        if not context_data:
            log.warning("LLM returned empty context data, skipping save")
            return False

        # Atomic write: write to .tmp then rename
        temp_path = cache_path.with_suffix(".tmp")
        with open(temp_path, 'wb') as f:
            f.write(context_data)

        # Rename is atomic on POSIX
        shutil.move(str(temp_path), str(cache_path))

        log.info(f"Context cache saved successfully for tool '{tool_name}'")
        return True
    except Exception as e:
        log.warning(f"Failed to save context cache for tool '{tool_name}': {e}")
        # Clean up temp file
        try:
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
        return False


def load_context_from_cache(
    llm: LLM, tool_name: str, cache_dir: Path, logger_instance: Optional[logging.Logger] = None
) -> bool:
    """
    Load LLM context from a cache file with validation.

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

        if cache_path.stat().st_size == 0:
            log.warning(f"Context cache file for '{tool_name}' is empty, skipping load")
            return False

        log.debug(f"Loading context from cache file: {cache_path}")

        try:
            with open(cache_path, 'rb') as f:
                context_data = f.read()
        except Exception as e:
            log.warning(f"Failed to read cache file: {e}")
            return False

        if not context_data:
            return False

        try:
            llm.load_context(context_data)
        except Exception as e:
            log.warning(f"LLM failed to load context data: {e}")
            log.warning(f"Cache file might be corrupted: {cache_path}")
            return False

        log.info(f"Context cache loaded successfully for tool '{tool_name}'")
        return True
    except Exception as e:
        log.warning(f"Failed to load context cache for tool '{tool_name}': {e}")
        return False


def add_to_context(
    llm: LLM, prompt: list, logger_instance: Optional[logging.Logger] = None
) -> bool:
    """
    Add content to the LLM context by generating a minimal response.

    This is a placeholder mechanism until official API support is available.
    It works by sending the prompt and generating a single token (which is discarded).
    It automatically appends an instruction to the prompt to minimize output.

    Args:
        llm (LLM): The LLM instance.
        prompt (list): The prompt messages to add to context.
        logger_instance (logging.Logger): Logger to use.

    Returns:
        bool: True if successful, False otherwise.
    """
    log = logger_instance or logger
    try:
        # Deep copy prompt to avoid modifying the original list
        import copy
        prompt_to_send = copy.deepcopy(prompt)

        # Add instruction to minimize response
        silence_instruction = " Respond with only a single space character."

        if prompt_to_send and isinstance(prompt_to_send, list):
            last_msg = prompt_to_send[-1]
            if "content" in last_msg:
                content = last_msg["content"]
                if isinstance(content, list):
                    # Handle content list (e.g. [{"type": "text", "text": ...}])
                    text_found = False
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            part["text"] = part.get("text", "") + silence_instruction
                            text_found = True
                            break
                    if not text_found:
                        # If no text part found, append one
                        content.append({"type": "text", "text": silence_instruction})
                elif isinstance(content, str):
                    last_msg["content"] += silence_instruction

        # Generate a single token to add the prompt to context
        generated_tokens = []
        for token in llm.generate(prompt=prompt_to_send, max_generated_tokens=1):
            generated_tokens.append(token)

        # Log debug info
        generated_text = "".join(generated_tokens)
        log.debug("Context addition generated token: %s", repr(generated_text))
        return True

    except Exception as e:
        log.warning("Failed to add to context: %s", e)
        return False

