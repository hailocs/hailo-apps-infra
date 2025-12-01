import argparse
from io import StringIO
from contextlib import redirect_stderr

from hailo_platform import VDevice
from hailo_platform.genai import LLM

from hailo_apps.python.core.common.defines import LLM_PROMPT_PREFIX, SHARED_VDEVICE_GROUP_ID, RESOURCES_MODELS_DIR_NAME, LLM_MODEL_NAME_H10
from hailo_apps.python.core.common.core import get_resource_path
from hailo_apps.python.core.gen_ai_utils.voice_processing.interaction import VoiceInteractionManager
from hailo_apps.python.core.gen_ai_utils.voice_processing.speech_to_text import SpeechToTextProcessor
from hailo_apps.python.core.gen_ai_utils.voice_processing.text_to_speech import TextToSpeechProcessor


class VoiceAssistantApp:
    """
    Manages the main application logic for the voice assistant.
    Builds the pipeline using common AI components.
    """

    def __init__(self, debug=False, no_tts=False):
        self.debug = debug
        self.no_tts = no_tts

        print("Initializing AI components... (This might take a moment)")

        # Suppress noisy ALSA messages during initialization
        with redirect_stderr(StringIO()):
            # 1. VDevice
            params = VDevice.create_params()
            params.group_id = SHARED_VDEVICE_GROUP_ID
            self.vdevice = VDevice(params)

            # 2. Speech to Text
            self.s2t = SpeechToTextProcessor(self.vdevice)

            # 3. LLM
            # USER CONFIGURATION: You can change the LLM model here.
            # By default, it uses the model defined in LLM_MODEL_NAME_H10.
            # To use a custom HEF, provide the absolute path to your .hef file.
            model_path = str(
                get_resource_path(
                    pipeline_name=None,
                    resource_type=RESOURCES_MODELS_DIR_NAME,
                    model=LLM_MODEL_NAME_H10,
                )
            )
            # Example of using a custom path:
            # model_path = "/path/to/your/custom_model.hef"

            self.llm = LLM(self.vdevice, model_path)
            self._recovery_seq = self.llm.get_generation_recovery_sequence()

            # 4. TTS
            self.tts = None
            if not no_tts:
                self.tts = TextToSpeechProcessor()

        print("✅ AI components ready!")

    def on_processing_start(self):
        if self.tts:
            self.tts.interrupt()

    def on_audio_ready(self, audio):
        # 1. Transcribe
        user_text = self.s2t.transcribe(audio)
        if not user_text:
            print("No speech detected.")
            return

        print(f"\nYou: {user_text}")
        print("\nLLM response:\n")

        # 2. Prepare TTS
        current_gen_id = None
        if self.tts:
            self.tts.clear_interruption()
            current_gen_id = self.tts.get_current_gen_id()

        # 3. Generate Response
        prompt = LLM_PROMPT_PREFIX + user_text

        # Format prompt as a list of messages for the LLM
        formatted_prompt = [{'role': 'user', 'content': prompt}]

        output = ''
        sentence_buffer = ''
        first_chunk_sent = False

        with self.llm.generate(prompt=formatted_prompt) as gen:
            for token in gen:
                if token == self._recovery_seq:
                    continue

                print(token, end='', flush=True)
                output += token

                if self.tts:
                    sentence_buffer += token
                    # Chunk and queue speech
                    sentence_buffer = self._chunk_and_queue_speech(
                        sentence_buffer, current_gen_id, not first_chunk_sent
                    )

                    if not first_chunk_sent and not self.tts.speech_queue.empty():
                        first_chunk_sent = True

        # 4. Send remaining text
        if self.tts and sentence_buffer.strip():
            self.tts.queue_text(sentence_buffer.strip(), current_gen_id)

        print()

    def _chunk_and_queue_speech(self, buffer, gen_id, is_first_chunk):
        # Use a comma as a delimiter only for the first chunk for faster response.
        delimiters = ['.', '?', '!']
        if is_first_chunk:
            delimiters.append(',')

        while True:
            # Find the first occurrence of any delimiter.
            positions = {buffer.find(d): d for d in delimiters if buffer.find(d) != -1}
            if not positions:
                break  # No delimiters found

            first_pos = min(positions.keys())
            chunk = buffer[:first_pos + 1]

            if chunk.strip():
                self.tts.queue_text(chunk.strip(), gen_id)

            buffer = buffer[first_pos + 1:]

        return buffer

    def on_clear_context(self):
        self.llm.clear_context()
        print("Context cleared.")

    def close(self):
        if self.tts:
            self.tts.stop()

        # Clean up LLM resources
        try:
            self.llm.release()
        except Exception:
            pass

        # We rely on process cleanup for VDevice release or Python's GC,
        # matching original pattern.
        # Ideally self.vdevice.release() but it's shared.


def main():
    parser = argparse.ArgumentParser(
        description='A simple, voice-controlled AI assistant for your terminal.')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode to save recorded audio files.')
    parser.add_argument('--no-tts', action='store_true',
                        help='Disable text-to-speech output for lower resource usage.')

    args = parser.parse_args()

    if args.debug:
        print("Debug mode enabled: Audio will be saved to 'debug_audio_*.wav' files.")
    if args.no_tts:
        print("TTS disabled: Running in low-resource mode.")

    # Initialize the app
    app = VoiceAssistantApp(debug=args.debug, no_tts=args.no_tts)

    # Initialize the interaction manager
    interaction = VoiceInteractionManager(
        title="Voice Assistant",
        on_audio_ready=app.on_audio_ready,
        on_processing_start=app.on_processing_start,
        on_clear_context=app.on_clear_context,
        on_shutdown=app.close,
        debug=args.debug
    )

    interaction.run()


if __name__ == "__main__":
    main()
