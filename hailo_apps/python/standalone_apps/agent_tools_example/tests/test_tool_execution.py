"""
Unit tests for tool execution module.
"""

import unittest
from unittest.mock import Mock
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import tool_execution

class TestToolExecution(unittest.TestCase):
    def setUp(self):
        self.mock_runner = Mock(return_value={"ok": True, "result": "success"})
        self.tools_lookup = {
            "test_tool": {
                "name": "test_tool",
                "runner": self.mock_runner
            }
        }

    def test_execute_valid_tool(self):
        """Test executing a valid tool call."""
        call = {"name": "test_tool", "arguments": {"arg": "value"}}
        result = tool_execution.execute_tool_call(call, self.tools_lookup)

        self.assertTrue(result["ok"])
        self.assertEqual(result["result"], "success")
        self.mock_runner.assert_called_once_with({"arg": "value"})

    def test_execute_unknown_tool(self):
        """Test executing a tool that doesn't exist."""
        call = {"name": "unknown_tool", "arguments": {}}
        result = tool_execution.execute_tool_call(call, self.tools_lookup)

        self.assertFalse(result["ok"])
        self.assertIn("Unknown tool", result["error"])

    def test_execute_tool_exception(self):
        """Test handling exceptions raised by tool runner."""
        self.mock_runner.side_effect = ValueError("Test error")
        call = {"name": "test_tool", "arguments": {}}
        result = tool_execution.execute_tool_call(call, self.tools_lookup)

        self.assertFalse(result["ok"])
        self.assertIn("Test error", result["error"])

    def test_invalid_call_format(self):
        """Test handling invalid tool call format."""
        result = tool_execution.execute_tool_call("not a dict", self.tools_lookup) # type: ignore
        self.assertFalse(result["ok"])
        self.assertIn("Invalid tool call format", result["error"])

if __name__ == "__main__":
    unittest.main()

