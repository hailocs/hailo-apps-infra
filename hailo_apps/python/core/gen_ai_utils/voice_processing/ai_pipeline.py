"""
AI Pipeline module for Hailo Voice Assistant.

This module orchestrates Speech-to-Text, LLM, and Text-to-Speech components.
"""

import queue
from typing import Optional

import numpy as np
from hailo_platform import VDevice

from hailo_apps.python.core.common.defines import LLM_PROMPT_PREFIX, SHARED_VDEVICE_GROUP_ID
from hailo_apps.python.core.gen_ai_utils.voice_processing.llm_processor import LLMProcessor
from hailo_apps.python.core.gen_ai_utils.voice_processing.speech_to_text import (
    SpeechToTextProcessor,
)
from hailo_apps.python.core.gen_ai_utils.voice_processing.text_to_speech import (
    TextToSpeechProcessor,
)


class AIPipeline:
    """
    Manages the AI pipeline from speech-to-text, to a large language model,
    and finally to text-to-speech.

    This class handles the complexities of streaming responses and ensuring that
    new user interactions can gracefully interrupt and replace ongoing ones.
    """

    def __init__(self, no_tts: bool = False):
        """
        Initializes all components of the AI pipeline.

        Args:
            no_tts (bool): If True, disables text-to-speech output.
        """
        self.no_tts = no_tts
        self._setup_hailo_ai()

        self.tts: Optional[TextToSpeechProcessor] = None
        if not no_tts:
            self.tts = TextToSpeechProcessor()

    def _setup_hailo_ai(self):
        """Initializes Hailo AI platform components (VDevice, S2T, LLM)."""
        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        self._vdevice = VDevice(params)
        self.speech2text = SpeechToTextProcessor(self._vdevice)
        self.llm = LLMProcessor(self._vdevice)

    def interrupt(self):
        """
        Interrupts any ongoing speech.
        """
        if self.tts:
            self.tts.interrupt()

    def process(self, audio: np.ndarray) -> str:
        """
        Processes recorded audio to generate and speak a response.

        Args:
            audio (np.ndarray): The raw audio data from the microphone.

        Returns:
            str: The generated text response from the language model.
        """
        # 1. Prepare for the new generation
        current_gen_id = None
        if self.tts:
            self.tts.clear_interruption()
            current_gen_id = self.tts.get_current_gen_id()

        # 2. Transcribe
        user_text = self.speech2text.transcribe(audio)
        if not user_text:
            return ""

        print("\nLLM response:\n")

        # 3. Get response from LLM
        prompt = LLM_PROMPT_PREFIX + user_text

        output = ''
        sentence_buffer = ''
        first_chunk_sent = False

        for token in self.llm.generate(prompt):
            print(token, end='', flush=True)
            output += token

            if self.tts:
                sentence_buffer += token
                # 4. Chunk and queue speech
                sentence_buffer = self._chunk_and_queue_speech(
                    sentence_buffer, current_gen_id, not first_chunk_sent
                )

                if not first_chunk_sent and not self.tts.speech_queue.empty():
                    first_chunk_sent = True

        # 5. Send remaining text
        if self.tts and sentence_buffer.strip():
            self.tts.queue_text(sentence_buffer.strip(), current_gen_id)

        print()
        return output

    def _chunk_and_queue_speech(
        self, buffer: str, gen_id: Optional[int], is_first_chunk: bool
    ) -> str:
        """
        Chunks a buffer of text into sentences and adds them to the speech queue.

        Args:
            buffer (str): The text buffer to be chunked.
            gen_id (Optional[int]): The current generation ID.
            is_first_chunk (bool): If true, also uses commas as delimiters.

        Returns:
            str: The remaining text in the buffer after chunking.
        """
        if not self.tts:
            return buffer

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

    def close(self):
        """Release resources."""
        if self.tts:
            self.tts.stop()
        # VDevice release is handled by the platform when needed, or could be explicit here
        # self._vdevice.release()

