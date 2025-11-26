"""
Voice-enabled Interactive CLI chat agent.

Combines voice input/output capabilities with the tool-using chat agent.

Usage:
  python -m hailo_apps.hailo_app_python.tools.voice_chat_agent
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
from pathlib import Path

from hailo_platform import VDevice
from hailo_platform.genai import LLM

from hailo_apps.python.core.gen_ai_utils.voice_processing.audio_recorder import AudioRecorder
from hailo_apps.python.core.gen_ai_utils.voice_processing.speech_to_text import SpeechToTextProcessor
from hailo_apps.python.core.gen_ai_utils.voice_processing.text_to_speech import TextToSpeechProcessor
from hailo_apps.python.core.gen_ai_utils.llm_utils import (
    message_formatter,
    streaming,
    context_manager,
)
from hailo_apps.python.core.common.defines import SHARED_VDEVICE_GROUP_ID

try:
    from . import (
        agent_utils,
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
    import agent_utils
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
        self.need_system_prompt = True # Initialize default state

        # Initialize tools lookup
        self.tools_lookup = {self.selected_tool_name: selected_tool}

        # Initialize recorder
        try:
            self.recorder = AudioRecorder(debug=debug)
        except Exception as e:
            print(f"[Error] Failed to initialize audio recorder: {e}")
            raise

        self.is_recording = False
        self.lock = threading.Lock()

        print("Initializing AI components...")

        # Initialize Hailo VDevice and Models
        try:
            params = VDevice.create_params()
            params.group_id = SHARED_VDEVICE_GROUP_ID
            self.vdevice = VDevice(params)
        except Exception as e:
            print(f"[Error] Failed to create VDevice: {e}")
            raise

        # LLM
        try:
            self.llm = LLM(self.vdevice, hef_path)
        except Exception as e:
            print(f"[Error] Failed to initialize LLM: {e}")
            self.vdevice.release()
            raise

        # S2T
        try:
            self.s2t = SpeechToTextProcessor(self.vdevice)
        except Exception as e:
            print(f"[Error] Failed to initialize Speech-to-Text: {e}")
            self.llm.release()
            self.vdevice.release()
            raise

        # TTS
        self.tts = None
        if not no_tts:
            try:
                self.tts = TextToSpeechProcessor()
            except Exception as e:
                print(f"[Warning] Failed to initialize TTS: {e}")
                print("Continuing without TTS support.")

        # Initialize Context
        self._init_context()

        print("✅ AI components ready!")

    def _init_context(self):
        system_text = system_prompt.create_system_prompt([self.selected_tool])
        logger.debug("SYSTEM PROMPT:\n%s", system_text)

        cache_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        try:
            context_loaded = context_manager.load_context_from_cache(self.llm, self.selected_tool_name, cache_dir, logger)
        except Exception as e:
            logger.warning("Failed to load context cache: %s", e)
            context_loaded = False

        if not context_loaded:
            logger.info("Initializing system prompt...")
            try:
                prompt = [message_formatter.messages_system(system_text)]
                context_manager.add_to_context(self.llm, prompt, logger)
                context_manager.save_context_to_cache(self.llm, self.selected_tool_name, cache_dir, logger)
                self.need_system_prompt = False
            except Exception as e:
                logger.error("Failed to initialize system context: %s", e)
                # Fallback: try to send system prompt with first message
                self.need_system_prompt = True
        else:
            logger.info("Loaded cached context.")
            self.need_system_prompt = False

        self.system_text = system_text

    def toggle_recording(self):
        with self.lock:
            if not self.is_recording:
                self.start_recording()
            else:
                self.stop_recording()

    def start_recording(self):
        if self.tts:
            try:
                self.tts.interrupt()
            except Exception as e:
                logger.warning("Failed to interrupt TTS: %s", e)

        try:
            self.recorder.start()
            self.is_recording = True
            print("\n🔴 Recording started. Press SPACE to stop.")
        except Exception as e:
            print(f"[Error] Failed to start recording: {e}")
            self.is_recording = False

    def stop_recording(self):
        print("\nProcessing... Please wait.")
        try:
            audio = self.recorder.stop()
        except Exception as e:
            print(f"[Error] Failed to stop recording: {e}")
            self.is_recording = False
            return

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
        try:
            user_text = self.s2t.transcribe(audio)
        except Exception as e:
            print(f"[Error] Transcription failed: {e}")
            return

        if not user_text:
            print("No speech detected.")
            return

        print(f"\nYou: {user_text}")

        # 2. Process with LLM and Tools
        self.process_interaction(user_text)

    def process_interaction(self, user_text):
        # Check context limits
        # Reason: Proactive trimming to avoid hitting token limits during response generation
        try:
            context_cleared = context_manager.check_and_trim_context(self.llm, logger_instance=logger)
            if context_cleared:
                self.need_system_prompt = True
        except Exception as e:
            logger.warning("Context check failed: %s", e)

        # Prepare prompt
        if self.need_system_prompt:
            # Reason: Re-insert system prompt if context was cleared to maintain tool instructions
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
            # Reason: Clear any pending speech to respond to new input immediately
            self.tts.clear_interruption()
            current_gen_id = self.tts.get_current_gen_id()

        # Reason: Filter output to hide XML tags from user while accumulating them for parsing
        token_filter = streaming.StreamingTextFilter(debug_mode=self.debug)

        # Generator loop
        try:
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
        except Exception as e:
            print(f"\n[Error] LLM generation failed: {e}")
            logger.error("LLM generation error: %s", traceback.format_exc())
            return

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
        context_cleared = agent_utils.update_context_with_tool_result(
            self.llm, result, self.system_text, user_text, logger
        )

        # update_context_with_tool_result already handles rebuilding context if needed,
        # but we might want to track need_system_prompt state if something fails?
        # The shared util does not modify 'self.need_system_prompt'.
        # If context was cleared and rebuilt inside util, we don't need to do anything.
        # If context was NOT cleared, we don't need to do anything.
        # So we just need to ensure 'self.need_system_prompt' is consistent?
        # Actually, the util handles the LLM context update.
        # If it returns True (context cleared), it means it rebuilt it with system prompt.
        # So we can assume system prompt is in context now.
        if context_cleared:
            self.need_system_prompt = False

    def close(self):
        print("\nShutting down...")
        if self.is_recording:
            try:
                self.recorder.stop()
            except Exception:
                pass

        if self.tts:
            try:
                self.tts.stop()
            except Exception:
                pass

        if hasattr(self, 'recorder'):
            try:
                self.recorder.close()
            except Exception:
                pass

        # Cleanup shared resources (LLM, VDevice, Tool)
        tool_module = self.selected_tool.get("module")
        agent_utils.cleanup_resources(
            getattr(self, 'llm', None),
            getattr(self, 'vdevice', None),
            tool_module,
            logger
        )


def main():
    config.setup_logging()

    # Validate configuration
    try:
        config.validate_config()
    except ValueError as e:
        print(f"[Configuration Error] {e}")
        return

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
    try:
        modules = tool_discovery.discover_tool_modules()
        all_tools = tool_discovery.collect_tools(modules)
    except Exception as e:
        print(f"[Error] Failed to discover tools: {e}")
        logger.debug(traceback.format_exc())
        return

    if not all_tools:
        print("No tools found.")
        return

    tool_thread, tool_result = tool_selection.start_tool_selection_thread(all_tools)
    selected_tool = tool_selection.get_tool_selection_result(tool_thread, tool_result)

    if not selected_tool:
        return

    tool_execution.initialize_tool_if_needed(selected_tool)

    # Start App
    try:
        app = VoiceAgentApp(hef_path, selected_tool, debug=args.debug, no_tts=args.no_tts)
    except Exception:
        # Error already printed in __init__
        return

    TerminalUI.show_banner(
        title="Voice-Enabled Tool Agent",
        controls={
            "SPACE": "start/stop recording",
            "Q": "quit",
            "C": "clear context",
        }
    )

    while True:
        try:
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
                    try:
                        app.llm.clear_context()
                        print("Context cleared.")

                        # Try to reload cached context after clearing
                        cache_dir = Path(os.path.dirname(os.path.abspath(__file__)))
                        context_reloaded = context_manager.load_context_from_cache(
                            app.llm, app.selected_tool_name, cache_dir, logger
                        )
                        if context_reloaded:
                            app.need_system_prompt = False
                            logger.info("Context reloaded from cache after clear")
                        else:
                            app.need_system_prompt = True
                            logger.info("No cache available after clear, will reinitialize on next message")
                    except Exception as e:
                        print(f"[Error] Failed to clear context: {e}")
                        app.need_system_prompt = True
        except KeyboardInterrupt:
            app.close()
            break
        except Exception as e:
            print(f"[Error] Unexpected error in main loop: {e}")
            logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
