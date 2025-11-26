
import unittest
from unittest.mock import MagicMock, patch
import logging
from hailo_apps.python.core.gen_ai_utils.llm_utils.context_manager import (
    add_to_context,
)

class TestContextManager(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_logger = MagicMock()

    def test_add_to_context_success(self):
        # Setup mock to yield one token
        self.mock_llm.generate.return_value = iter([" "])

        prompt = [{"role": "system", "content": "test"}]
        result = add_to_context(self.mock_llm, prompt, self.mock_logger)

        self.assertTrue(result)

        # Check that prompt was modified in the call to generate
        expected_prompt = [{"role": "system", "content": "test Respond with only a single space character."}]
        self.mock_llm.generate.assert_called_once_with(prompt=expected_prompt, max_generated_tokens=1)
        self.mock_logger.debug.assert_called()

    def test_add_to_context_content_list(self):
        # Setup mock
        self.mock_llm.generate.return_value = iter([" "])

        # Prompt with content as list (e.g. multimodal)
        prompt = [{
            "role": "user",
            "content": [
                {"type": "image", "image": "data"},
                {"type": "text", "text": "Describe this"}
            ]
        }]

        result = add_to_context(self.mock_llm, prompt, self.mock_logger)
        self.assertTrue(result)

        # Verify the text part was modified
        args, kwargs = self.mock_llm.generate.call_args
        actual_prompt = kwargs['prompt']
        text_content = actual_prompt[0]['content'][1]['text']
        self.assertEqual(text_content, "Describe this Respond with only a single space character.")

    def test_add_to_context_failure(self):
        # Setup mock to raise exception
        self.mock_llm.generate.side_effect = Exception("LLM Error")

        prompt = [{"role": "system", "content": "test"}]
        result = add_to_context(self.mock_llm, prompt, self.mock_logger)

        self.assertFalse(result)
        self.mock_logger.warning.assert_called_with("Failed to add to context: LLM Error")

if __name__ == "__main__":
    unittest.main()
