import pytest
from unittest.mock import MagicMock, patch, ANY
import numpy as np

# Mock hailo_platform and piper before importing AIPipeline
hailo_mock = MagicMock()
piper_mock = MagicMock()

import sys
sys.modules['hailo_platform'] = hailo_mock
sys.modules['piper'] = piper_mock

from processing import AIPipeline

@pytest.fixture
def mock_dependencies():
    """Fixture to provide fresh mocks for each test."""
    with patch('processing.VDevice', new=MagicMock()) as mock_vdevice, \
         patch('processing.Speech2Text', new=MagicMock()) as mock_s2t, \
         patch('processing.LLM', new=MagicMock()) as mock_llm, \
         patch('processing.PiperVoice', new=MagicMock()) as mock_piper:
        
        # Mock the context manager for LLM generation
        mock_llm.return_value.generate.return_value.__enter__.return_value = iter(['hello', ' world'])
        
        yield {
            "vdevice": mock_vdevice,
            "s2t": mock_s2t,
            "llm": mock_llm,
            "piper": mock_piper
        }

def test_pipeline_initialization(mock_dependencies):
    """Test that the AIPipeline initializes its components correctly."""
    pipeline = AIPipeline()
    mock_dependencies['vdevice'].assert_called_once()
    mock_dependencies['s2t'].assert_called_once()
    mock_dependencies['llm'].assert_called_once()
    mock_dependencies['piper'].load.assert_called_once()
    assert pipeline.speech_thread.is_alive()

def test_interrupt(mock_dependencies):
    """Test the interrupt method."""
    pipeline = AIPipeline()
    mock_process = MagicMock()
    pipeline.current_speech_process = mock_process
    pipeline.speech_queue.put((0, "test"))

    pipeline.interrupt()

    assert pipeline._interrupted.is_set()
    assert pipeline.generation_id == 1
    mock_process.kill.assert_called_once()
    assert pipeline.current_speech_process is None
    assert pipeline.speech_queue.empty()

def test_process(mock_dependencies):
    """Test the main process method."""
    pipeline = AIPipeline()
    
    # Mock the return value of speech-to-text
    mock_segment = MagicMock()
    mock_segment.text = "this is a test"
    mock_dependencies['s2t'].return_value.generate_all_segments.return_value = [mock_segment]

    # Mock the LLM to return some tokens
    mock_dependencies['llm'].return_value.generate.return_value.__enter__.return_value = iter(['Here', ' is', ' a', ' response', '.'])

    audio_data = np.zeros(16000, dtype=np.float32)
    result = pipeline.process(audio_data)

    mock_dependencies['s2t'].return_value.generate_all_segments.assert_called_once()
    mock_dependencies['llm'].return_value.generate.assert_called_once()
    
    # Check that text was added to the speech queue
    assert not pipeline.speech_queue.empty()
    gen_id, text = pipeline.speech_queue.get()
    assert gen_id == 0
    assert text == "Here is a response."

    assert result == "Here is a response."
