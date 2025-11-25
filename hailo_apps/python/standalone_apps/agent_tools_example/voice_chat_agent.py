"""
Voice-enabled Interactive CLI chat agent.

Combines voice input/output capabilities with the tool-using chat agent.

Usage:
  python -m hailo_apps.hailo_app_python.tools.voice_chat_agent
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from io import StringIO
from contextlib import redirect_stderr
from pathlib import Path

from hailo_platform import VDevice
from hailo_platform.genai import LLM

from hailo_apps.python.core.gen_ai_utils.voice_processing.audio_recorder import AudioRecorder
from hailo_apps.python.core.gen_ai_utils.voice_processing.speech_to_text import SpeechToTextProcessor
from hailo_apps.python.core.gen_ai_utils.voice_processing.text_to_speech import TextToSpeechProcessor
from hailo_apps.python.core.gen_ai_utils.llm_utils import (
    context_manager,
    message_formatter,
    streaming,
)
from hailo_apps.python.core.common.defines import SHARED_VDEVICE_GROUP_ID

try:
    from . import (
        config,
        system_prompt,
        text_processing,
        tool_discovery,
        tool_execution,
        tool_selection,
    )
except ImportError:
    # Add the script's directory to sys.path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import config
    import system_prompt
    import text_processing
    import tool_discovery
    import tool_execution
    import tool_selection

from hailo_apps.python.core.gen_ai_utils.llm_utils.terminal_ui import TerminalUI

logger = config.LOGGER


class VoiceAgentApp:
    def __init__(self, hef_path, selected_tool, debug=False, no_tts=False):
        self.debug = debug
        self.no_tts = no_tts
        self.selected_tool = selected_tool
        self.selected_tool_name = selected_tool.get("name", "")

        # Initialize tools lookup
        self.tools_lookup = {self.selected_tool_name: selected_tool}

        # Initialize recorder
        self.recorder = AudioRecorder(debug=debug)
        self.is_recording = False
        self.lock = threading.Lock()

        print("Initializing AI components...")

        # Initialize Hailo VDevice and Models
        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        self.vdevice = VDevice(params)

        # LLM
        self.llm = LLM(self.vdevice, hef_path)

        # S2T
        self.s2t = SpeechToTextProcessor(self.vdevice)

        # TTS
        self.tts = None
        if not no_tts:
            self.tts = TextToSpeechProcessor()

        # Initialize Context
        self._init_context()

        print("✅ AI components ready!")

    def _init_context(self):
        system_text = system_prompt.create_system_prompt([self.selected_tool])
        logger.debug("SYSTEM PROMPT:\n%s", system_text)

        cache_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        context_loaded = context_manager.load_context_from_cache(self.llm, self.selected_tool_name, cache_dir, logger)

        self.need_system_prompt = False
        if not context_loaded:
            logger.info("Initializing system prompt...")
            context_manager.initialize_system_prompt_context(self.llm, system_text, logger)
            context_manager.save_context_to_cache(self.llm, self.selected_tool_name, cache_dir, logger)
        else:
            logger.info("Loaded cached context.")

        self.system_text = system_text

    def toggle_recording(self):
        with self.lock:
            if not self.is_recording:
                self.start_recording()
            else:
                self.stop_recording()

    def start_recording(self):
        if self.tts:
            self.tts.interrupt()
        self.recorder.start()
        self.is_recording = True
        print("\n🔴 Recording started. Press SPACE to stop.")

    def stop_recording(self):
        print("\nProcessing... Please wait.")
        audio = self.recorder.stop()
        self.is_recording = False

        if audio.size > 0:
            self.process_audio(audio)
        else:
            print("No audio recorded.")

        TerminalUI.show_banner(
            title="Voice-Enabled Tool Agent",
            controls={
                "SPACE": "start/stop recording",
                "Q": "quit",
                "C": "clear context",
            }
        )

    def process_audio(self, audio):
        # 1. Transcribe
        user_text = self.s2t.transcribe(audio)
        if not user_text:
            print("No speech detected.")
            return

        print(f"\nYou: {user_text}")

        # 2. Process with LLM and Tools
        self.process_interaction(user_text)

    def process_interaction(self, user_text):
        # Check context limits
        context_cleared = context_manager.check_and_trim_context(self.llm, logger_instance=logger)
        if context_cleared:
            self.need_system_prompt = True

        # Prepare prompt
        if self.need_system_prompt:
            prompt = [
                message_formatter.messages_system(self.system_text),
                message_formatter.messages_user(user_text),
            ]
            self.need_system_prompt = False
        else:
            prompt = [message_formatter.messages_user(user_text)]

        # Generate response
        print("Assistant: ", end="", flush=True)

        full_response = ""
        sentence_buffer = ""
        first_chunk_sent = False

        current_gen_id = None
        if self.tts:
            self.tts.clear_interruption()
            current_gen_id = self.tts.get_current_gen_id()

        token_filter = streaming.StreamingTextFilter(debug_mode=self.debug)

        # Generator loop
        for token in self.llm.generate(prompt, temperature=config.TEMPERATURE):
            full_response += token
            cleaned = token_filter.process_token(token)

            if cleaned:
                print(cleaned, end="", flush=True)
                if self.tts:
                    sentence_buffer += cleaned
                    # Chunk speech
                    # Simple chunking logic from AI pipeline
                    delimiters = ['.', '?', '!']
                    if not first_chunk_sent:
                        delimiters.append(',')

                    while True:
                        positions = {sentence_buffer.find(d): d for d in delimiters if sentence_buffer.find(d) != -1}
                        if not positions:
                            break

                        first_pos = min(positions.keys())
                        chunk = sentence_buffer[:first_pos + 1]

                        if chunk.strip():
                            self.tts.queue_text(chunk.strip(), current_gen_id)
                            if not first_chunk_sent:
                                first_chunk_sent = True

                        sentence_buffer = sentence_buffer[first_pos + 1:]

        # Flush remaining speech
        remaining_text = token_filter.get_remaining()
        if remaining_text:
            print(remaining_text, end="", flush=True)
            if self.tts:
                sentence_buffer += remaining_text

        if self.tts and sentence_buffer.strip():
            self.tts.queue_text(sentence_buffer.strip(), current_gen_id)

        print()

        # Check for tool calls
        tool_call = text_processing.parse_function_call(full_response)
        if tool_call:
            self.handle_tool_call(tool_call, user_text)

    def handle_tool_call(self, tool_call, user_text):
        # Execute tool
        result = tool_execution.execute_tool_call(tool_call, self.tools_lookup)
        tool_execution.print_tool_result(result)

        if self.tts:
            if result.get("ok"):
                res_str = str(result.get("result", ""))
                # Speak result if it's short enough, or a summary
                if len(res_str) < 200:
                    self.tts.queue_text(f"The result is {res_str}")
                else:
                    self.tts.queue_text("I have calculated the result.")
            else:
                self.tts.queue_text("There was an error executing the tool.")

        # Add result to context
        tool_result_text = json.dumps(result, ensure_ascii=False)
        tool_response_message = f"<tool_response>{tool_result_text}</tool_response>"

        context_cleared = context_manager.check_and_trim_context(self.llm, logger_instance=logger)
        if context_cleared:
            # Rebuild context if cleared
            prompt = [
                message_formatter.messages_system(self.system_text),
                message_formatter.messages_user(user_text),
                message_formatter.messages_user(tool_response_message),
            ]
        else:
            prompt = [message_formatter.messages_user(tool_response_message)]

        # Update context
        try:
            for _ in self.llm.generate(prompt, max_generated_tokens=1):
                break
        except Exception as e:
            logger.debug(f"Context update failed: {e}")

    def close(self):
        print("\nShutting down...")
        if self.is_recording:
            self.recorder.stop()
        if self.tts:
            self.tts.stop()
        if hasattr(self, 'recorder'):
            self.recorder.close()
        if hasattr(self, 'llm'):
            self.llm.release()
        if hasattr(self, 'vdevice'):
            self.vdevice.release()

        # Cleanup tool resources
        tool_module = self.selected_tool.get("module")
        if tool_module and hasattr(tool_module, "cleanup_tool"):
            try:
                tool_module.cleanup_tool()
            except Exception:
                pass


def main():
    config.setup_logging()

    parser = argparse.ArgumentParser(description='Voice-enabled AI Tool Agent')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-tts', action='store_true', help='Disable TTS')
    args = parser.parse_args()

    # Get HEF
    try:
        hef_path = config.get_hef_path()
    except ValueError as e:
        print(f"[Error] {e}")
        return

    if not os.path.exists(hef_path):
        print("[Error] HEF file not found.")
        return

    # Tool Selection
    modules = tool_discovery.discover_tool_modules()
    all_tools = tool_discovery.collect_tools(modules)
    if not all_tools:
        print("No tools found.")
        return

    tool_thread, tool_result = tool_selection.start_tool_selection_thread(all_tools)
    selected_tool = tool_selection.get_tool_selection_result(tool_thread, tool_result)

    if not selected_tool:
        return

    tool_execution.initialize_tool_if_needed(selected_tool)

    # Start App
    app = VoiceAgentApp(hef_path, selected_tool, debug=args.debug, no_tts=args.no_tts)
    TerminalUI.show_banner(
        title="Voice-Enabled Tool Agent",
        controls={
            "SPACE": "start/stop recording",
            "Q": "quit",
            "C": "clear context",
        }
    )

    while True:
        ch = TerminalUI.get_char().lower()
        if ch == "q":
            app.close()
            break
        elif ch == " ":
            app.toggle_recording()
        elif ch == "\x03":
            app.close()
            break
        elif ch == "c":
            if app.llm:
                app.llm.clear_context()
                print("Context cleared.")


if __name__ == "__main__":
    main()

