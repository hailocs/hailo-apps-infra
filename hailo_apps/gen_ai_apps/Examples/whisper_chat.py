from hailo_platform import VDevice
from hailo_platform.genai import Speech2Text
from hailo_apps.hailo_app_python.core.common.core import get_resource_path
from hailo_apps.hailo_app_python.core.common.defines import RESOURCES_MODELS_DIR_NAME, WHISPER_MODEL_NAME_H10
import librosa

vdevice = None
whisper = None

try:
    vdevice = VDevice()
    whisper = Speech2Text(vdevice, get_resource_path(resource_type=RESOURCES_MODELS_DIR_NAME, model=WHISPER_MODEL_NAME_H10))
    
    # Load audio file
    audio_path = 'audio.wav'
    audio_data, sample_rate = librosa.load(audio_path, sr=None)
    
    if audio_data is None or len(audio_data) == 0:
        raise ValueError("Could not load audio file or audio file is empty")
    
    # Generate transcription
    segments = whisper.generate_all_segments(audio_data, timeout_ms=15000)
    
    if segments and len(segments) > 0:
        transcription = segments[0].text.strip()
        print(transcription)
    else:
        print("No transcription generated")
    
except Exception as e:
    print(f"Error occurred: {e}")
    
finally:
    # Clean up resources
    if whisper:
        try:
            whisper.release()
        except Exception as e:
            print(f"Error releasing Speech2Text: {e}")
    
    if vdevice:
        try:
            vdevice.release()
        except Exception as e:
            print(f"Error releasing VDevice: {e}")