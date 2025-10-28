import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from recorder import Recorder

# Mock the pyaudio library
pyaudio_mock = MagicMock()
mock_stream = MagicMock()
mock_stream.start_stream = MagicMock()
mock_stream.stop_stream = MagicMock()
mock_stream.close = MagicMock()
pyaudio_mock.PyAudio.return_value.open.return_value = mock_stream
pyaudio_mock.PyAudio.return_value.terminate = MagicMock()

@patch('recorder.pyaudio', pyaudio_mock)
def test_recorder_initialization():
    """Test that the Recorder initializes correctly."""
    recorder = Recorder(debug=True)
    assert recorder.debug
    assert not recorder.is_recording
    assert recorder.audio_frames == []

@patch('recorder.pyaudio', pyaudio_mock)
def test_recorder_start():
    """Test the start method of the Recorder."""
    recorder = Recorder()
    recorder.start()
    assert recorder.is_recording
    pyaudio_mock.PyAudio.return_value.open.assert_called_once()
    mock_stream.start_stream.assert_called_once()

@patch('recorder.pyaudio', pyaudio_mock)
def test_recorder_stop_no_frames():
    """Test the stop method with no audio frames."""
    recorder = Recorder()
    recorder.start()
    audio = recorder.stop()
    assert not recorder.is_recording
    mock_stream.stop_stream.assert_called_once()
    mock_stream.close.assert_called_once()
    assert isinstance(audio, np.ndarray)
    assert audio.size == 0

@patch('recorder.pyaudio', pyaudio_mock)
def test_recorder_stop_with_frames():
    """Test the stop method with audio frames."""
    recorder = Recorder()
    recorder.start()
    # Simulate adding audio frames
    raw_audio = (np.sin(np.linspace(0, 440 * 2 * np.pi, 16000)) * 32767).astype(np.int16)
    recorder.audio_frames.append(raw_audio.tobytes())
    
    audio = recorder.stop()
    assert not recorder.is_recording
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.size > 0
    # Check that the audio is roughly in the range [-1, 1]
    assert np.max(np.abs(audio)) <= 1.0

@patch('recorder.pyaudio', pyaudio_mock)
def test_recorder_close():
    """Test the close method."""
    recorder = Recorder()
    recorder.close()
    pyaudio_mock.PyAudio.return_value.terminate.assert_called_once()

@patch('recorder.pyaudio', pyaudio_mock)
def test_callback():
    """Test the PyAudio callback."""
    recorder = Recorder()
    recorder.is_recording = True
    test_data = b'test_audio_data'
    result = recorder.callback(test_data, 0, 0, 0)
    assert recorder.audio_frames == [test_data]
    assert result == (test_data, pyaudio_mock.paContinue)

    # Test when not recording
    recorder.is_recording = False
    recorder.audio_frames = []
    recorder.callback(test_data, 0, 0, 0)
    assert recorder.audio_frames == []
