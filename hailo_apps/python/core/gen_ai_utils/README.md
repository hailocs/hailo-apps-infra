# Hailo GenAI Utilities

This package provides shared utilities for Generative AI applications on Hailo platforms. It encapsulates common functionality for voice processing, LLM interaction, and context management.

## Architecture

The package is organized into the following modules:

- **`voice_processing`**: Handles audio input/output and speech processing.
- **`llm_utils`**: Provides utilities for managing Large Language Model interactions and terminal UI.

## Modules

### Voice Processing (`voice_processing`)

Provides components for building voice-enabled applications.

- **`AudioRecorder`**: Handles microphone recording using PyAudio.
- **`SpeechToTextProcessor`**: Wraps Hailo's Speech2Text API (Whisper).
- **`LLMProcessor`**: Wraps Hailo's LLM API for text generation.
- **`TextToSpeechProcessor`**: Handles speech synthesis using Piper TTS.
- **`AIPipeline`**: Orchestrates the flow between S2T, LLM, and TTS.

#### Usage Example

```python
from hailo_apps.python.core.gen_ai_utils.voice_processing.ai_pipeline import AIPipeline

# Initialize pipeline
pipeline = AIPipeline(no_tts=False)

# Process audio
response_text = pipeline.process(audio_data)
```

### LLM Utilities (`llm_utils`)

Provides helpers for building LLM agents.

- **`context_manager`**: Manages LLM context window, including tracking usage and caching.
- **`message_formatter`**: Helper functions to format messages for the LLM.
- **`streaming`**: Utilities for streaming LLM responses and filtering special tokens.
- **`terminal_ui`**: Terminal UI helpers for interactive applications.

#### Usage Example

```python
from hailo_apps.python.core.gen_ai_utils.llm_utils import context_manager

# Check if context needs trimming
context_manager.check_and_trim_context(llm_instance)

# Terminal UI
from hailo_apps.python.core.gen_ai_utils.llm_utils.terminal_ui import TerminalUI

# Show a custom banner
TerminalUI.show_banner(
    title="My Voice App",
    controls={
        "SPACE": "start/stop recording",
        "Q": "quit",
    }
)

# Read a single character
ch = TerminalUI.get_char()
```

## Installation Requirements

### Voice Processing Module

If you plan to use voice processing features (TTS), you must install the Piper TTS model.

**See [Voice Processing Module Documentation](voice_processing/README.md) for installation instructions.**

## Integration Guide

To use these utilities in your standalone application:

1. Import the desired modules from `hailo_apps.python.core.gen_ai_utils`.
2. Ensure `hailo_apps` is in your Python path.
3. If using voice processing, install the Piper TTS model (see above).
4. Follow the usage examples above.

