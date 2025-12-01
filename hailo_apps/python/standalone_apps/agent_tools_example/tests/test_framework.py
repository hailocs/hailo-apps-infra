"""
Test framework utilities for Agent Tools Example.

Provides mocks and fixtures for testing agent components without hardware.
"""

import sys
import os
from unittest.mock import MagicMock, Mock

# Add parent directory to path to allow importing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class MockLLM:
    """Mock LLM for testing."""
    def __init__(self, responses=None):
        self.responses = responses or []
        self.context = []
        self.capacity = 4096

    def generate(self, prompt, **kwargs):
        """Yield tokens from pre-defined responses."""
        if not self.responses:
            yield ""
            return

        response = self.responses.pop(0)
        # Simulate token streaming
        for char in response:
            yield char

    def max_context_capacity(self):
        return self.capacity

    def get_context_usage_size(self):
        return len(str(self.context))

    def clear_context(self):
        self.context = []

    def save_context(self):
        return b"mock_context_data"

    def load_context(self, data):
        pass

    def release(self):
        pass

class MockVDevice:
    """Mock VDevice for testing."""
    def release(self):
        pass

    @staticmethod
    def create_params():
        return Mock()

def get_mock_tool():
    """Return a simple mock tool definition."""
    return {
        "name": "mock_tool",
        "display_description": "A mock tool for testing",
        "llm_description": "Use this tool for testing",
        "tool_def": {
            "type": "function",
            "function": {
                "name": "mock_tool",
                "description": "Use this tool for testing",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        "runner": Mock(return_value={"ok": True, "result": "success"}),
        "module": Mock()
    }

