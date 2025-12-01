import threading
import traceback
from typing import Callable, Optional

import numpy as np

from hailo_apps.python.core.gen_ai_utils.llm_utils.terminal_ui import TerminalUI
from hailo_apps.python.core.gen_ai_utils.voice_processing.audio_recorder import AudioRecorder


class VoiceInteractionManager:
    """
    Manages the interactive voice loop (recording, UI, events).

    This class abstracts the common pattern of:
    1. Waiting for user input (SPACE to record, Q to quit, C to clear context).
    2. Managing the AudioRecorder.
    3. Delegating processing to callbacks.
    """

    def __init__(
        self,
        title: str,
        on_audio_ready: Callable[[np.ndarray], None],
        on_processing_start: Optional[Callable[[], None]] = None,
        on_clear_context: Optional[Callable[[], None]] = None,
        on_shutdown: Optional[Callable[[], None]] = None,
        debug: bool = False,
    ):
        """
        Args:
            title (str): Title for the terminal banner.
            on_audio_ready (Callable): Callback when recording finishes with audio data.
            on_processing_start (Callable): Callback when recording starts (e.g. to stop TTS).
            on_clear_context (Callable): Callback when 'C' is pressed.
            on_shutdown (Callable): Callback when 'Q' or Ctrl+C is pressed.
            debug (bool): Enable debug logging for recorder.
        """
        self.title = title
        self.on_audio_ready = on_audio_ready
        self.on_processing_start = on_processing_start
        self.on_clear_context = on_clear_context
        self.on_shutdown = on_shutdown
        self.debug = debug

        self.recorder = AudioRecorder(debug=debug)
        self.is_recording = False
        self.lock = threading.Lock()

        self.controls = {
            "SPACE": "start/stop recording",
            "Q": "quit",
            "C": "clear context",
        }

    def run(self):
        """Starts the main interaction loop."""
        TerminalUI.show_banner(title=self.title, controls=self.controls)

        try:
            while True:
                ch = TerminalUI.get_char().lower()
                if ch == "q":
                    self.close()
                    break
                elif ch == " ":
                    self.toggle_recording()
                elif ch == "\x03":  # Ctrl+C
                    self.close()
                    break
                elif ch == "c":
                    if self.on_clear_context:
                        self.on_clear_context()
        except KeyboardInterrupt:
            self.close()
        except Exception as e:
            print(f"[Error] Unexpected error in main loop: {e}")
            if self.debug:
                traceback.print_exc()

    def toggle_recording(self):
        with self.lock:
            if not self.is_recording:
                self.start_recording()
            else:
                self.stop_recording()

    def start_recording(self):
        if self.on_processing_start:
            try:
                self.on_processing_start()
            except Exception as e:
                print(f"[Warning] Failed to execute start callback: {e}")

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
            if self.on_audio_ready:
                self.on_audio_ready(audio)
        else:
            print("No audio recorded.")

        TerminalUI.show_banner(title=self.title, controls=self.controls)

    def close(self):
        print("\nShutting down...")
        if self.is_recording:
            try:
                self.recorder.stop()
            except Exception:
                pass

        try:
            self.recorder.close()
        except Exception:
            pass

        if self.on_shutdown:
            try:
                self.on_shutdown()
            except Exception as e:
                print(f"[Error] Failed during shutdown callback: {e}")

