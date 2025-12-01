"""
Unit tests for text processing module.
"""

import unittest
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import text_processing

class TestTextProcessing(unittest.TestCase):
    def test_parse_valid_xml_json(self):
        """Test parsing valid XML wrapped JSON."""
        response = '<tool_call>{"name": "test", "arguments": {"a": 1}}</tool_call>'
        result = text_processing.parse_function_call(response)

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["arguments"]["a"], 1)

    def test_parse_nested_json_string(self):
        """Test parsing when arguments are a stringified JSON."""
        response = '<tool_call>{"name": "test", "arguments": "{\\"a\\": 1}"}</tool_call>'
        result = text_processing.parse_function_call(response)

        self.assertIsNotNone(result)
        self.assertEqual(result["arguments"]["a"], 1)

    def test_parse_single_quotes(self):
        """Test parsing JSON with single quotes (common LLM error)."""
        response = "<tool_call>{'name': 'test', 'arguments': {'a': 1}}</tool_call>"
        result = text_processing.parse_function_call(response)

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "test")

    def test_parse_missing_closing_tag(self):
        """Test parsing with missing closing tag (truncated response)."""
        response = '<tool_call>{"name": "test", "arguments": {"a": 1}}'
        result = text_processing.parse_function_call(response)

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "test")

    def test_parse_no_tool_call(self):
        """Test parsing text without tool call."""
        response = "Just some text response."
        result = text_processing.parse_function_call(response)
        self.assertIsNone(result)

    def test_parse_malformed_json(self):
        """Test parsing completely broken JSON."""
        response = '<tool_call>{"name": "test", broken...</tool_call>'
        result = text_processing.parse_function_call(response)
        self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()

