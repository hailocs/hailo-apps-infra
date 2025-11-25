"""
Text processing utilities for tool parsing.
"""

import json
from typing import Any, Dict, Optional


def parse_function_call(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse function call from LLM response.

    ONLY supports XML-wrapped format:
    <tool_call>
    {"name": "...", "arguments": {...}}
    </tool_call>

    Args:
        response: Raw response string from LLM

    Returns:
        Parsed function call dict with 'name' and 'arguments' keys, or None if not found
    """
    def validate_and_fix_call(call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate that call has required fields and fix nested JSON."""
        if not isinstance(call, dict):
            return None
        # Must have 'name' field
        if "name" not in call or not call.get("name"):
            return None
        # Must have 'arguments' field
        if "arguments" not in call:
            return None
        # Fix nested JSON in arguments
        if isinstance(call.get("arguments"), str):
            try:
                call["arguments"] = json.loads(call["arguments"])  # nested JSON fix
            except Exception:
                pass
        # Ensure arguments is a dict
        if not isinstance(call.get("arguments"), dict):
            return None
        return call

    # ONLY support XML-wrapped function call
    if "<tool_call>" in response:
        try:
            start = response.find("<tool_call>") + len("<tool_call>")
            # Find closing tag, or use brace matching if missing
            end = response.find("</tool_call>", start)
            if end == -1:
                # No closing tag, use brace matching
                json_str = response[start:].strip()
                # Find the complete JSON object by matching braces
                brace_count = 0
                json_end = -1
                for i, char in enumerate(json_str):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                if json_end > 0:
                    json_str = json_str[:json_end]
                else:
                    return None
            else:
                json_str = response[start:end].strip()

            json_str = json_str.replace("'", '"')
            call = json.loads(json_str)
            return validate_and_fix_call(call)
        except Exception:
            return None

    return None

