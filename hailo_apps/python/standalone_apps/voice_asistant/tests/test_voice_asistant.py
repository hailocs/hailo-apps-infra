import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

# Mock 'piper' and 'piper.voice' before importing app modules
mock_piper = MagicMock()
sys.modules['piper'] = mock_piper
sys.modules['piper.voice'] = MagicMock()

from hailo_apps.python.standalone_apps.voice_asistant.voice_asistant import VoiceAssistantApp

class TestVoiceAssistantApp(unittest.TestCase):
    def setUp(self):
        # Patch external dependencies
        self.vdevice_patcher = patch('hailo_apps.python.standalone_apps.voice_asistant.voice_asistant.VDevice')
        self.llm_patcher = patch('hailo_apps.python.standalone_apps.voice_asistant.voice_asistant.LLM')
        self.s2t_patcher = patch('hailo_apps.python.standalone_apps.voice_asistant.voice_asistant.SpeechToTextProcessor')
        self.tts_patcher = patch('hailo_apps.python.standalone_apps.voice_asistant.voice_asistant.TextToSpeechProcessor')
        self.get_resource_path_patcher = patch('hailo_apps.python.standalone_apps.voice_asistant.voice_asistant.get_resource_path')

        self.mock_vdevice = self.vdevice_patcher.start()
        self.mock_llm = self.llm_patcher.start()
        self.mock_s2t = self.s2t_patcher.start()
        self.mock_tts = self.tts_patcher.start()
        self.mock_get_resource_path = self.get_resource_path_patcher.start()

        # Setup mocks
        self.mock_get_resource_path.return_value = "/fake/path/to/model.hef"
        self.mock_llm_instance = self.mock_llm.return_value
        self.mock_llm_instance.get_generation_recovery_sequence.return_value = None
        self.mock_s2t_instance = self.mock_s2t.return_value
        self.mock_tts_instance = self.mock_tts.return_value

    def tearDown(self):
        self.vdevice_patcher.stop()
        self.llm_patcher.stop()
        self.s2t_patcher.stop()
        self.tts_patcher.stop()
        self.get_resource_path_patcher.stop()

    def test_initialization(self):
        """Test that the app initializes components correctly."""
        app = VoiceAssistantApp(debug=False, no_tts=False)

        self.mock_vdevice.assert_called()
        self.mock_s2t.assert_called_with(self.mock_vdevice.return_value)
        self.mock_llm.assert_called_with(self.mock_vdevice.return_value, "/fake/path/to/model.hef")
        self.mock_tts.assert_called()

    def test_initialization_no_tts(self):
        """Test initialization without TTS."""
        app = VoiceAssistantApp(debug=False, no_tts=True)

        self.mock_tts.assert_not_called()
        self.assertIsNone(app.tts)

    def test_on_audio_ready_flow(self):
        """Test the main processing flow."""
        app = VoiceAssistantApp()

        # Mock inputs
        audio_data = b"fake_audio"
        self.mock_s2t_instance.transcribe.return_value = "Hello AI"

        # Mock LLM generation
        # The generate method returns a context manager that yields a generator
        mock_generator = iter(["Hi", " there", "."])
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_generator
        mock_context_manager.__exit__.return_value = None
        self.mock_llm_instance.generate.return_value = mock_context_manager

        # Mock TTS queue
        self.mock_tts_instance.speech_queue.empty.return_value = True

        # Run
        app.on_audio_ready(audio_data)

        # Assertions
        self.mock_s2t_instance.transcribe.assert_called_with(audio_data)

        # Verify LLM called with formatted prompt
        # Note: We check if generate was called. The exact prompt might depend on constants imports.
        self.mock_llm_instance.generate.assert_called()
        call_args = self.mock_llm_instance.generate.call_args
        self.assertEqual(call_args.kwargs['prompt'][0]['role'], 'user')
        self.assertIn("Hello AI", call_args.kwargs['prompt'][0]['content'])

        # Verify TTS was called
        self.assertTrue(self.mock_tts_instance.queue_text.called)

    def test_close(self):
        """Test cleanup."""
        app = VoiceAssistantApp()
        app.close()

        self.mock_tts_instance.stop.assert_called()
        self.mock_llm_instance.release.assert_called()

if __name__ == '__main__':
    unittest.main()
