"""
Configuration file for the ASR Demo application.

This file centralizes all the settings and parameters that can be customized
by the user, making it easier to tweak the application's behavior without
modifying the core logic.
"""

# --- Audio Recording Settings ---
TARGET_SR = 16000  # Target sample rate for audio recording (in Hz)
CHUNK_SIZE = 1024      # Number of frames per buffer

# --- AI Model Paths ---
# These paths point to the model files used in the AI pipeline.
# Make sure these files are available in the specified locations.
WHISPER_HEF_PATH = "Whisper-Base.hef"
LLM_HEF_PATH = "Qwen2.5-Coder-1.5B-Instruct.hef"
TTS_ONNX_PATH = "en_US-amy-low.onnx"

# --- Text-to-Speech (TTS) Settings ---
# These settings control the characteristics of the synthesized speech.
# Feel free to experiment with these values to change the voice.
TTS_VOLUME = 0.8          # Volume (0.0 to 1.0)
TTS_LENGTH_SCALE = 0.6    # Speech rate (lower is faster)
TTS_NOISE_SCALE = 0.6     # Voice variability (lower is more consistent)
# Pronunciation variability (lower is more consistent)
TTS_W_SCALE = 0.6

# --- LLM Settings ---
# This prefix is added to the user's transcribed speech before sending it
# to the language model. It helps guide the model's response.
LLM_PROMPT_PREFIX = "Respond in up to three sentences. "

# --- Temporary File Settings ---
# Directory to store temporary audio files.
TEMP_WAV_DIR = "/tmp"
