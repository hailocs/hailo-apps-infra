"""
Tool discovery module.

Handles automatic discovery and collection of tool modules.
"""

import importlib
import pkgutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, List, Dict


def discover_tool_modules() -> List[ModuleType]:
    """
    Discover tool modules from files named 'tool_*.py' in the tools directory.

    Returns:
        List of imported tool modules
    """
    modules: List[ModuleType] = []
    current_dir = Path(__file__).parent

    # Ensure current directory is in sys.path (works from any directory)
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    # Build module name: use package prefix if available, otherwise just module name
    package_prefix = f"{__package__}." if __package__ else ""

    for module_info in pkgutil.iter_modules([str(current_dir)]):
        if not module_info.name.startswith("tool_"):
            continue
        try:
            module_name = f"{package_prefix}{module_info.name}"
            modules.append(importlib.import_module(module_name))
        except Exception:
            continue
    return modules


def collect_tools(modules: List[ModuleType]) -> List[Dict[str, Any]]:
    """
    Collect tool metadata and schemas from tool modules.

    Args:
        modules: List of tool modules to process

    Returns:
        List of dictionaries with keys:
            - name: Tool name (string)
            - display_description: User-facing description for CLI (string)
            - llm_description: Description for LLM/tool schema (string)
            - tool_def: Full tool definition dict following the TOOL_SCHEMA format
            - runner: Callable that executes the tool (usually module.run)
            - module: The originating module (for debugging/logging)
    """
    tools: List[Dict[str, Any]] = []
    seen_names: set[str] = set()
    for m in modules:
        run_fn = getattr(m, "run", None)
        # Skip template tool to avoid confusing the model
        module_name = getattr(m, "name", None)
        if module_name == "template_tool":
            continue

        tool_schemas = getattr(m, "TOOLS_SCHEMA", None)
        display_description = getattr(m, "display_description", None)
        llm_description_attr = getattr(m, "description", None)

        if tool_schemas and isinstance(tool_schemas, list):
            for entry in tool_schemas:
                if not isinstance(entry, dict):
                    continue
                if entry.get("type") != "function":
                    continue
                function_def = entry.get("function", {})
                name = function_def.get("name")
                description = function_def.get("description", llm_description_attr or "")
                if not name or not callable(run_fn):
                    continue
                if name in seen_names:
                    continue
                seen_names.add(name)
                display_desc = display_description if display_description else description or name
                tools.append(
                    {
                        "name": str(name),
                        "display_description": str(display_desc),
                        "llm_description": str(description),
                        "tool_def": entry,
                        "runner": run_fn,
                        "module": m,
                    }
                )
            continue

        # Legacy fallback: build schema on the fly if TOOLS_SCHEMA not provided
        name = getattr(m, "name", None)
        llm_description = llm_description_attr
        schema = getattr(m, "schema", None)
        if name and llm_description and callable(run_fn):
            if name in seen_names:
                continue
            seen_names.add(name)
            display_desc = display_description if display_description else llm_description
            parameters = schema if isinstance(schema, dict) else {"type": "object", "properties": {}}
            tool_def = {
                "type": "function",
                "function": {
                    "name": str(name),
                    "description": str(llm_description),
                    "parameters": parameters,
                },
            }
            tools.append(
                {
                    "name": str(name),
                    "display_description": str(display_desc),
                    "llm_description": str(llm_description),
                    "tool_def": tool_def,
                    "runner": run_fn,
                    "module": m,
                }
            )
    tools.sort(key=lambda t: t["name"])
    return tools

