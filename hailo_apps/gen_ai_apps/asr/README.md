# Terminal Voice Assistant

A simple, voice-controlled AI assistant that runs in your terminal. This project uses the Hailo AI platform to perform speech-to-text, generate a response with a large language model, and speak the response back to you using text-to-speech.

It's designed to be a starting point for makers and developers who want to build their own voice-controlled AI applications. The code is structured to be easy to understand and modify.

## Features

- **Voice-Activated**: Start and stop recordings with a simple key press.
- **Real-Time Interaction**: The assistant processes your speech and responds vocally.
- **Streaming Response**: The AI's response is synthesized and played back in chunks, allowing for faster, more natural-sounding interactions.
- **Interruptible**: You can interrupt the assistant at any time by starting a new recording.
- **Easy to Customize**: A central `config.py` file lets you easily change AI models, voice characteristics, and other parameters without touching the core logic.
- **Debug Mode**: Save your recordings to WAV files to analyze microphone quality and troubleshoot issues.

## Requirements

### System Dependencies
- A Linux-based OS (tested on Ubuntu/Debian)
- Python 3
- `portaudio19-dev`
- `python3-dev`
- `alsa-utils`

### Hailo AI Platform Dependencies
You must have the following Hailo package installed:
- `hailort=5.1.0`

**Installation instructions:**
- **Raspberry Pi**: Install via RPi apt server (recommended)
- **Other platforms**: Download the wheel file manually from Hailo's official repository and install with `pip install <wheel-file>`

For detailed installation instructions, refer to the official Hailo documentation.

### Python Packages
The specific Python packages required are listed in `requirements.txt` and will be installed by the setup script. They include:
- `numpy`
- `pyaudio`
- `librosa`
- `piper-tts`
- `soundfile`
- `scipy`

## Installation

The `setup.sh` script is provided to automate the installation process.

### Prerequisites
1. **Install Hailo dependencies** (see requirements section above)
2. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

3. **Make the setup script executable:**
   ```bash
   chmod +x setup.sh
   ```

### Setup Options

#### Default Setup (Recommended for RPi users)
The setup script defaults to using system-site-packages, which will use your RPi apt installation.
```bash
./setup.sh
```

This is recommended for:
- **Raspberry Pi** (Hailo packages are pre-installed via apt)
- **Other platforms** with system-wide Hailo wheel installation

#### Clean Virtual Environment (Advanced)
If you want a completely isolated environment:
```bash
./setup.sh --no-system-site-packages
# Then manually install Hailo packages in the venv:
source venv_asr/bin/activate
pip install <path-to-hailort-wheel-file>
```
Then manually install Hailo packages in the venv.

### What the Setup Script Does
The script will:
- Check for required system dependencies
- Verify Hailo packages are available
- Create a Python virtual environment (`venv_asr`)
- Install the required Python packages
- Download the Text-to-Speech voice model and AI model files

### Getting Help
Run `./setup.sh --help` to see all available options and detailed usage instructions.

## Usage

1.  **Activate the virtual environment:**
    ```bash
    source venv_asr/bin/activate
    ```

2.  **Run the application:**
    ```bash
    python3 main.py
    ```

3.  **Controls:**
    - Press the **SPACE** bar to start and stop recording.
    - Press **Q** or **Ctrl+C** to quit the application. The application will perform a graceful shutdown, stopping any audio playback and releasing all resources.

### Debug Mode

To run in debug mode, which saves your recorded audio to `debug_audio_*.wav` files, use the `--debug` flag:
```python3 main.py --debug
```

## Configuration

The `config.py` file allows you to easily customize the application's behavior. Here are some of the key settings you can change:

- **`TARGET_SR`**: The sample rate for audio recording. **Note:** This should not be changed, as the AI models are trained for a specific sample rate (16000 Hz).
- **`CHUNK_SIZE`**: The buffer size for audio frames.
- **`WHISPER_HEF_PATH`**: Path to the Whisper model file for speech-to-text.
- **`LLM_HEF_PATH`**: Path to the Large Language Model file.
- **`TTS_ONNX_PATH`**: Path to the Piper TTS voice model.
- **TTS Settings**: You can adjust `TTS_VOLUME`, `TTS_LENGTH_SCALE` (speech speed), and other parameters to change the voice.
- **`LLM_PROMPT_PREFIX`**: A prefix added to your speech to guide the LLM's response.

## Troubleshooting

If you are getting poor transcription results from the speech-to-text model, it is often due to microphone quality or background noise.

1.  **Check Your Microphone Quality**: Run the application in debug mode to save your recordings:
    ```bash
    python3 main.py --debug
    ```
    This will create `debug_audio_*.wav` files in the project directory. Listen to these files to check if your voice is clear and free of significant background noise.

2.  **Ensure Good Recording Quality**: For best results, use a decent quality microphone and try to be in a quiet environment.

3.  **Do Not Change `TARGET_SR`**: The `TARGET_SR` in `config.py` is set to 16000 Hz, which is the sample rate the Whisper model was trained on. Changing this value will result in poor transcription performance.

## Running Tests

To run the test suite:

1.  **Activate the virtual environment:**
    ```bash
    source venv_asr/bin/activate
    ```

2.  **Run the tests:**
    ```bash
    python3 -m pytest
    ```

The test suite includes 16 tests that verify all core functionality including recording, processing, and graceful shutdown.

## Project Structure

```
asr_demo/
├── main.py              # Main application entry point
├── processing.py        # AI pipeline (S2T, LLM, TTS)
├── recorder.py          # Audio recording and processing
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
├── setup.sh            # Installation script
├── README.md           # This file
├── tests/              # Unit tests
│   ├── test_main.py
│   ├── test_processing.py
│   └── test_recorder.py
└── venv_asr/           # Virtual environment (created by setup)
```

## Contributing

This project is designed to be a starting point for makers. Feel free to:
- Modify the configuration in `config.py`
- Add new features to the existing modules
- Create your own voice-controlled applications based on this code
- Submit issues or pull requests for improvements

## License

This project is open source and available under the MIT License.
