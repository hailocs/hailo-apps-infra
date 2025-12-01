import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..')))

# We need to mock piper because text_to_speech imports it
from unittest.mock import MagicMock
sys.modules['piper'] = MagicMock()
sys.modules['piper.voice'] = MagicMock()

from hailo_apps.python.core.gen_ai_utils.voice_processing.text_to_speech import clean_text_for_tts

class TestTextCleaning(unittest.TestCase):
    def test_clean_markdown_bold_italic(self):
        """Test removing bold and italic markers."""
        self.assertEqual(clean_text_for_tts("Hello **world**"), "Hello world")
        self.assertEqual(clean_text_for_tts("This is *italic*"), "This is italic")
        self.assertEqual(clean_text_for_tts("Mixed ***bold italic***"), "Mixed bold italic")
        self.assertEqual(clean_text_for_tts("__Underlined__ text"), "Underlined text")

    def test_clean_markdown_code(self):
        """Test removing code backticks."""
        self.assertEqual(clean_text_for_tts("Use `print()` function"), "Use print() function")
        self.assertEqual(clean_text_for_tts("```python\ncode\n```"), "python code")

    def test_clean_markdown_headers(self):
        """Test removing headers."""
        self.assertEqual(clean_text_for_tts("# Title"), "Title")
        self.assertEqual(clean_text_for_tts("## Subtitle"), "Subtitle")
        # My regex was ^#+\s*. It depends on multiline flag.
        # Let's check if input is multiline.
        self.assertEqual(clean_text_for_tts("## Header\nContent"), "Header Content")

    def test_clean_markdown_links(self):
        """Test removing links."""
        self.assertEqual(clean_text_for_tts("Click [here](http://example.com)"), "Click here")

    def test_clean_special_symbols(self):
        """Test removing noisy symbols."""
        self.assertEqual(clean_text_for_tts("User@Host"), "User Host")
        self.assertEqual(clean_text_for_tts("Value ~ 10"), "Value 10")
        self.assertEqual(clean_text_for_tts("Tags #hash"), "Tags hash") # # was not in the removal list?
        # Wait, I didn't put # in the removal regex explicitly in clean_text_for_tts comments, but I put it in headers.
        # Hash often used for comments or headers.
        # Let's check the implementation.
        # Implementation: text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE) for headers.
        # But inline hash? e.g. "Number #1". This might be kept or stripped.
        # I did NOT add # to the generic symbol strip list: re.sub(r"[~@^|\\<>{}\[\]]", " ", text)
        # Markdown strip removed * and _.
        # So # might remain if not at start of line.
        # TTS saying "Number hash one" or "Number pound one" is okay-ish, but "hash hash hash" is noise.
        # Let's see what happens.
        pass

    def test_clean_brackets(self):
        """Test removing brackets which are often metadata."""
        # I added [] to symbol strip list.
        self.assertEqual(clean_text_for_tts("Data [hidden]"), "Data hidden")
        # Wait, regex was [~@^|\\<>{}\[\]]. So [ and ] are replaced by space.
        # "Data [hidden]" -> "Data  hidden " -> "Data hidden" (normalized)

    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        self.assertEqual(clean_text_for_tts("Hello   world\n\nTest"), "Hello world Test")

if __name__ == '__main__':
    unittest.main()

