from hailo_platform import VDevice
from hailo_platform.genai import Speech2Text
import librosa

vdevice = VDevice()
whisper = Speech2Text(vdevice, 'Whisper-Base.hef')
print(whisper.generate_all_segments(librosa.load('audio.wav', sr=None)[0], timeout_ms=15000)[0].text.strip())
whisper.release()
vdevice.release()