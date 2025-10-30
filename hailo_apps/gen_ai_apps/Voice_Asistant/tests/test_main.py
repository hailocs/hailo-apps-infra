import pytest
from unittest.mock import MagicMock, patch

# Mock dependencies before importing the main module
@pytest.fixture(autouse=True)
def mock_main_dependencies():
    with patch('main.Recorder') as mock_recorder, \
         patch('main.AIPipeline') as mock_pipeline, \
         patch('main.termios', MagicMock()), \
         patch('main.tty', MagicMock()), \
         patch('main.sys', MagicMock()):
        yield {
            'recorder': mock_recorder,
            'pipeline': mock_pipeline
        }

from main import TerminalRecorderApp

def test_app_initialization(mock_main_dependencies):
    """Test the initialization of the TerminalRecorderApp."""
    app = TerminalRecorderApp(debug=True)
    mock_main_dependencies['recorder'].assert_called_with(debug=True)
    mock_main_dependencies['pipeline'].assert_called_once()
    assert app.debug
    assert not app.is_recording

def test_start_recording(mock_main_dependencies):
    """Test the start_recording method."""
    app = TerminalRecorderApp()
    app.start_recording()
    app.ai_pipeline.interrupt.assert_called_once()
    app.recorder.start.assert_called_once()
    assert app.is_recording

def test_stop_recording_with_audio(mock_main_dependencies):
    """Test the stop_recording method when audio is recorded."""
    app = TerminalRecorderApp()
    app.is_recording = True
    
    # Mock that the recorder returns some audio
    mock_audio = MagicMock()
    mock_audio.size = 100
    app.recorder.stop.return_value = mock_audio

    app.stop_recording()

    app.recorder.stop.assert_called_once()
    assert not app.is_recording
    app.ai_pipeline.process.assert_called_with(mock_audio)

def test_stop_recording_no_audio(mock_main_dependencies):
    """Test the stop_recording method when no audio is recorded."""
    app = TerminalRecorderApp()
    app.is_recording = True
    
    # Mock that the recorder returns no audio
    mock_audio = MagicMock()
    mock_audio.size = 0
    app.recorder.stop.return_value = mock_audio

    app.stop_recording()

    app.recorder.stop.assert_called_once()
    assert not app.is_recording
    app.ai_pipeline.process.assert_not_called()

def test_toggle_recording(mock_main_dependencies):
    """Test the toggle_recording method."""
    app = TerminalRecorderApp()
    
    # Mock the audio return value for stop_recording
    mock_audio = MagicMock()
    mock_audio.size = 100
    app.recorder.stop.return_value = mock_audio
    
    # Test starting
    app.toggle_recording()
    assert app.is_recording
    app.recorder.start.assert_called_once()

    # Test stopping
    app.toggle_recording()
    assert not app.is_recording
    app.recorder.stop.assert_called_once()

def test_close_while_recording(mock_main_dependencies):
    """Test the close method while a recording is in progress."""
    app = TerminalRecorderApp()
    app.is_recording = True

    app.close()

    app.recorder.stop.assert_called_once()
    app.ai_pipeline.interrupt.assert_called_once()
    app.recorder.close.assert_called_once()
    assert not app.is_recording

def test_close_when_idle(mock_main_dependencies):
    """Test the close method when the app is idle."""
    app = TerminalRecorderApp()
    app.is_recording = False

    app.close()

    app.recorder.stop.assert_not_called() # Should not be called if not recording
    app.ai_pipeline.interrupt.assert_called_once()
    app.recorder.close.assert_called_once()
