import threading
import argparse
import signal
import cv2
import sys
from backend import Backend
import json
import concurrent.futures
import os
import time
import subprocess
import select
import termios
import tty
os.environ["QT_QPA_PLATFORM"] = "xcb"

class App:
    def __init__(self, prompts=None, camera_id=0):
        self.camera_id = camera_id
        self.running = True
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.kill_hailo_processes()  # Kill any processes using /dev/hailo0 before initializing backend
        self.backend = Backend(prompts=prompts)
        signal.signal(signal.SIGINT, self.signal_handler)

        # Interactive mode variables
        self.interactive_mode = False
        self.frozen_frame = None
        self.hailo_processing_enabled = True
        self.waiting_for_question = False
        self.waiting_for_continue = False
        self.user_question = ""

    def signal_handler(self, sig, frame):
        self.stop()

    def stop(self):
        self.running = False
        self.interactive_mode = False
        # Don't call cv2.destroyAllWindows() from signal handler
        if self.backend:
            self.backend.close()
        self.executor.shutdown(wait=True)

    def check_keyboard_input(self):
        """Check for Enter key press in a non-blocking way"""
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            input_line = sys.stdin.readline().strip()
            return input_line
        return None

    def get_user_input_nonblocking(self):
        """Get user text input in a non-blocking way"""
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.readline().strip()
        return None

    def kill_camera_processes(self):
        """Kill processes that might be using the camera"""

        # List of common camera applications to kill
        camera_apps = ['cheese', 'guvcview', 'vlc', 'obs', 'skype', 'zoom', 'firefox', 'chrome', 'chromium']

        killed_processes = []

        # Try to find and kill processes using video devices
        try:
            # Check with lsof first
            for i in range(10):  # Check video0 through video9
                device = f'/dev/video{i}'
                if os.path.exists(device):
                    try:
                        result = subprocess.run(['lsof', device], capture_output=True, text=True)
                        if result.stdout:
                            # Parse lsof output to get PIDs
                            lines = result.stdout.strip().split('\n')[1:]  # Skip header
                            for line in lines:
                                parts = line.split()
                                if len(parts) > 1:
                                    pid = parts[1]
                                    process_name = parts[0]
                                    try:
                                        subprocess.run(['kill', '-9', pid], check=True)
                                        killed_processes.append(f"{process_name} (PID: {pid})")
                                    except subprocess.CalledProcessError:
                                        print(f"Failed to kill process {pid}")
                    except FileNotFoundError:
                        # lsof not available, try fuser
                        try:
                            result = subprocess.run(['fuser', '-k', device], capture_output=True, text=True)
                            if result.returncode == 0:
                                killed_processes.append(f"processes using {device}")
                        except FileNotFoundError:
                            pass
        except Exception as e:
            print(f"Error checking camera usage: {e}")

        # Kill common camera applications by name
        for app in camera_apps:
            try:
                result = subprocess.run(['pkill', '-f', app], capture_output=True, text=True)
                if result.returncode == 0:
                    killed_processes.append(app)
            except Exception:
                pass

        # Force reset camera module as last resort
        if killed_processes:
            try:
                subprocess.run(['sudo', 'rmmod', 'uvcvideo'], capture_output=True)
                time.sleep(1)
                subprocess.run(['sudo', 'modprobe', 'uvcvideo'], capture_output=True)
                time.sleep(2)
            except Exception as e:
                print(f"Could not reset camera module: {e}")

    def kill_hailo_processes(self):
        """Kill processes that might be using the Hailo device"""
        killed_processes = []

        try:
            # Check if /dev/hailo0 exists
            if os.path.exists('/dev/hailo0'):
                # Use fuser to find and kill processes using /dev/hailo0
                result = subprocess.run(['fuser', '/dev/hailo0'], capture_output=True, text=True)
                if result.stdout.strip():
                    pids = result.stdout.strip().split()
                    for pid in pids:
                        try:
                            subprocess.run(['kill', '-9', pid], check=True)
                            killed_processes.append(f"PID {pid}")
                        except subprocess.CalledProcessError:
                            # Try with sudo if regular kill fails
                            try:
                                subprocess.run(['sudo', 'kill', '-9', pid], check=True)
                                killed_processes.append(f"PID {pid} (sudo)")
                            except subprocess.CalledProcessError:
                                print(f"Failed to kill process {pid} even with sudo")
        except FileNotFoundError:
            # Alternative: use lsof if fuser is not available
            try:
                result = subprocess.run(['lsof', '/dev/hailo0'], capture_output=True, text=True)
                if result.stdout:
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    for line in lines:
                        parts = line.split()
                        if len(parts) > 1:
                            pid = parts[1]
                            process_name = parts[0]
                            try:
                                subprocess.run(['kill', '-9', pid], check=True)
                                killed_processes.append(f"{process_name} (PID: {pid})")
                            except subprocess.CalledProcessError:
                                print(f"Failed to kill process {pid}")
            except FileNotFoundError:
                print("Neither fuser nor lsof available for checking Hailo device usage")
        except Exception as e:
            print(f"Error checking Hailo device usage: {e}")

        if killed_processes:
            time.sleep(1)  # Give some time for processes to fully terminate

    def show_video(self):
        state = 'ready'

        # Kill any processes using the camera first
        self.kill_camera_processes()

        if self.camera_id != 0:
            cap = cv2.VideoCapture(self.camera_id)
        else:
            usb_devices = self.get_usb_video_devices()
            if not usb_devices:
                print("No USB video devices found.")
                return
            cap = cv2.VideoCapture(usb_devices[0])

        # camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print(f"Error: Could not open camera with ID {self.camera_id}")
            print("Trying alternative camera access methods...")

            # Try different camera IDs
            for cam_id in range(0, 5):
                print(f"Trying camera ID {cam_id}...")
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened():
                    print(f"Successfully opened camera ID {cam_id}")
                    break
                cap.release()

            if not cap.isOpened():
                print("Could not open any camera after killing processes")
                return

        hailo_future = None
        vlm_future = None

        # Print banner
        print("\n" + "="*80)
        print("  🎥  CAMERA STARTED  |  Press ENTER anytime to ask a question about the image")
        print("="*80 + "\n")

        while cap.isOpened() and self.running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera")
                break

            # Always show live video feed regardless of interactive mode
            cv2.imshow('Video', frame)

            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'):
                self.stop()
                break

            # Handle keyboard input based on current state
            user_input = self.check_keyboard_input()

            # State 1: Normal mode - check for Enter to start interactive mode
            if not self.interactive_mode and not self.waiting_for_question and not self.waiting_for_continue and user_input is not None:
                if user_input == "":  # Enter key pressed
                    self.interactive_mode = True
                    self.frozen_frame = frame.copy()
                    self.hailo_processing_enabled = False
                    self.waiting_for_question = True

                    # Show frozen frame in the existing Frame window
                    cv2.imshow('Frame', self.frozen_frame)
                    print("Type a question about the image (or press Enter for 'Describe the image'): ", end="", flush=True)

            # State 2: Waiting for user question
            elif self.waiting_for_question and user_input is not None:
                self.user_question = user_input
                self.waiting_for_question = False

                # Use default prompt if user just hits Enter without typing
                if not self.user_question.strip():
                    self.user_question = "Describe the image"
                    print(f"Using default prompt: '{self.user_question}'")

                print("Processing your question...")
                vlm_future = self.executor.submit(self.backend.vlm_custom_inference, self.frozen_frame.copy(), self.user_question)

            # State 3: Waiting for continue after VLM response
            elif self.waiting_for_continue and user_input is not None:
                if user_input == "":  # Enter key pressed
                    # Exit interactive mode and resume normal processing
                    self.interactive_mode = False
                    self.waiting_for_continue = False
                    self.hailo_processing_enabled = True
                    print("\n" + "="*80)
                    print("  ▶️  RESUMED  |  Press ENTER anytime to ask another question")
                    print("="*80 + "\n")

            # Handle VLM response when ready
            if self.interactive_mode and vlm_future and vlm_future.done() and not self.waiting_for_continue:
                try:
                    vlm_result = vlm_future.result()
                except Exception as e:
                    print(f"VLM error: {e}")

                vlm_future = None
                self.waiting_for_continue = True
                print("\nPress Enter to continue...")

            # Normal Hailo processing (when not in interactive mode)
            # Update the Frame window with current frame only when not in interactive mode
            if not self.interactive_mode and state == 'ready' and self.running and self.hailo_processing_enabled:
                hailo_future = self.executor.submit(self.backend.hailo_inference, frame.copy())
                state = 'processing'
                cv2.imshow('Frame', frame)

            elif not self.interactive_mode and state == 'processing' and hailo_future and hailo_future.done():
                try:
                    hailo_result = hailo_future.result()
                    current_time = time.strftime("%H:%M:%S")
                    answer = hailo_result.get('answer', '')

                    # Add icon based on event detection
                    if answer == "No Event Detected":
                        icon = "✅"  # Green check for normal state
                        display_text = f"{icon} {answer}"
                    elif answer.lower() in ["yes", "detected"]:
                        icon = "🚨"  # Alert for event detected
                        display_text = f"{icon} EVENT DETECTED!"
                    elif "error" in answer.lower() or "timeout" in answer.lower():
                        icon = "⚠️"  # Warning for errors
                        display_text = f"{icon} {answer}"
                    else:
                        icon = "🔔"  # Bell for other detected events
                        display_text = f"{icon} {answer}"

                    # Print in place using carriage return
                    print(f"\r\033[K[{current_time}] {display_text} | Time: {hailo_result.get('time', 'N/A')}", end="", flush=True)
                except Exception as e:
                    print(f"\r\033[K⚠️ Inference error: {e}")
                state = 'ready'
                hailo_future = None

        cap.release()
        cv2.destroyAllWindows()

    def get_usb_video_devices(self):
        """
        Get a list of video devices that are connected via USB and have video capture capability.
        """
        video_devices = [f'/dev/{device}' for device in os.listdir('/dev') if device.startswith('video')]
        usb_video_devices = []

        for device in video_devices:
            try:
                # Use udevadm to get detailed information about the device
                udevadm_cmd = ["udevadm", "info", "--query=all", "--name=" + device]
                result = subprocess.run(udevadm_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                output = result.stdout

                # Check if the device is connected via USB and has video capture capabilities
                if "ID_BUS=usb" in output and ":capture:" in output:
                    # Extract the number from /dev/videoX
                    device_number = int(device.split('video')[1])
                    usb_video_devices.append(device_number)
            except Exception as e:
                print(f"Error checking device {device}: {e}")

        return usb_video_devices

    def run(self):
        self.video_thread = threading.Thread(target=self.show_video)
        self.video_thread.start()
        try:
            self.video_thread.join()
        except KeyboardInterrupt:
            self.stop()
            self.video_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM App")
    parser.add_argument("--prompts", type=str, required=True, help="Path to a JSON file with prompts.")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID (default: 0 for first USB camera)")
    args = parser.parse_args()
    with open(args.prompts, "r") as f:
        prompts = json.load(f)
    app = App(prompts=prompts, camera_id=args.camera)
    app.run()
    sys.exit(0)