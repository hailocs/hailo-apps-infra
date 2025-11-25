"""
Tool execution module.

Handles tool initialization and execution.
"""

import json
import logging
from typing import Any, Dict

# Setup logger
logger = logging.getLogger(__name__)


def initialize_tool_if_needed(tool: Dict[str, Any]) -> None:
    """
    Initialize tool if it has an initialize_tool function.

    Args:
        tool: Tool dictionary containing a 'module' key.
    """
    tool_module = tool.get("module")
    if tool_module and hasattr(tool_module, "initialize_tool"):
        try:
            tool_module.initialize_tool()
        except Exception as e:
            logger.warning("Tool initialization failed: %s", e)


def execute_tool_call(
    tool_call: Dict[str, Any], tools_lookup: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Execute a tool call and return the result.

    Args:
        tool_call: Parsed tool call dictionary with 'name' and 'arguments' keys.
        tools_lookup: Dictionary mapping tool names to tool metadata.

    Returns:
        Tool execution result dictionary with 'ok' key and either 'result' or 'error'.
    """
    tool_name = str(tool_call.get("name", "")).strip()
    args = tool_call.get("arguments", {})
    logger.info("TOOL CALL: %s", tool_name)
    logger.debug("Tool call details - name: %s", tool_name)
    logger.debug("Tool call arguments:\n%s", json.dumps(args, indent=2, ensure_ascii=False))

    selected = tools_lookup.get(tool_name)
    if not selected:
        available = ", ".join(sorted(tools_lookup.keys()))
        logger.error(f"Unknown tool '{tool_name}'. Available: {available}")
        return {"ok": False, "error": f"Unknown tool '{tool_name}'. Available: {available}"}

    runner = selected.get("runner")
    if not callable(runner):
        logger.error(f"Tool '{tool_name}' is missing an executable runner.")
        return {"ok": False, "error": f"Tool '{tool_name}' is missing an executable runner."}

    try:
        result = runner(args)  # type: ignore[misc]
        logger.debug("TOOL EXECUTION RESULT:\n%s", json.dumps(result, indent=2, ensure_ascii=False))
        return result
    except Exception as exc:
        result = {"ok": False, "error": f"Tool raised exception: {exc}"}
        logger.error("Tool execution raised exception: %s", exc)
        logger.debug("Tool exception result:\n%s", json.dumps(result, indent=2, ensure_ascii=False))
        return result


def print_tool_result(result: Dict[str, Any]) -> None:
    """
    Print tool execution result to the user.

    Args:
        result: Tool execution result dictionary with 'ok' key.
    """
    if result.get("ok"):
        logger.info("Tool execution: SUCCESS")
        tool_result_text = result.get("result", "")
        if tool_result_text:
            print(f"\n[Tool] {tool_result_text}\n")
    else:
        logger.info("Tool execution: FAILED - %s", result.get("error", "Unknown error"))
        tool_error = result.get("error", "Unknown error")
        print(f"\n[Tool Error] {tool_error}\n")

