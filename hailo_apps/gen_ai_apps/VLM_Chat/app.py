import threading
import argparse
import signal
import cv2
import sys
from backend import Backend
import concurrent.futures
import os
import subprocess
import select
from pathlib import Path
os.environ["QT_QPA_PLATFORM"] = 'xcb'
# from hailo_apps.hailo_app_python.core.common.core import get_default_parser
# from hailo_apps.hailo_app_python.core.common.camera_utils import get_usb_video_devices
# from hailo_apps.hailo_app_python.core.common.defines import BASIC_PIPELINES_VIDEO_EXAMPLE_NAME, RESOURCES_ROOT_PATH_DEFAULT, RESOURCES_VIDEOS_DIR_NAME, RPI_NAME_I, USB_CAMERA

class App:
    def __init__(self, camera, cmaera_type):
        self.camera = camera
        self.camera_type = cmaera_type
        self.running = True
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.backend = Backend()
        signal.signal(signal.SIGINT, self.signal_handler)
        self.frozen_frame = None
        self.waiting_for_question = True
        self.waiting_for_continue = False
        self.user_question = ""

    def signal_handler(self, sig, frame):
        print('')
        self.running = False
        if self.backend:
            self.backend.close()
        self.executor.shutdown(wait=True)

    def check_keyboard_input(self):
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            input_line = sys.stdin.readline().strip()
            return input_line
        return None

    def get_user_input_nonblocking(self):
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.readline().strip()
        return None

    def show_video(self):
        usb_devices = self.get_usb_video_devices()
        if not usb_devices:
            print("No USB video devices found.")
            return
        cap = cv2.VideoCapture(usb_devices[0])
        # TODO
        # if self.camera_type == RPI_NAME_I:
        #     cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        # else:
        #     cap = cv2.VideoCapture(self.camera)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print("\n" + "="*80)
        print("  🎥  CAMERA STARTED  |  Ask a question about the image")
        print("="*80 + "\n")
        print("Type a question about the image (or press Enter for 'Describe the image'): ", end="", flush=True)
        vlm_future = None
        while cap.isOpened() and self.running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera")
                break

            cv2.imshow('Video', frame)

            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'):
                self.stop()
                break

            # Handle keyboard input
            user_input = self.check_keyboard_input()

            # Waiting for user question
            if self.waiting_for_question and user_input is not None:
                self.user_question = user_input
                self.waiting_for_question = False
                self.frozen_frame = frame.copy()

                # Use default prompt if user just hits Enter without typing
                if not self.user_question.strip():
                    self.user_question = "Describe the image"
                    print(f"Using default prompt: '{self.user_question}'")

                print("Processing your question...")
                vlm_future = self.executor.submit(self.backend.vlm_inference, self.frozen_frame.copy(), self.user_question)

            # Waiting for continue after VLM response
            elif self.waiting_for_continue and user_input is not None:
                if user_input == "":  # Enter key pressed
                    # Reset to waiting for next question
                    self.waiting_for_continue = False
                    self.waiting_for_question = True
                    print("\n" + "="*80)
                    print("  🎥  READY FOR NEXT QUESTION")
                    print("="*80 + "\n")
                    print("Type a question about the image (or press Enter for 'Describe the image'): ", end="", flush=True)

            # Handle VLM response when ready
            if vlm_future and vlm_future.done() and not self.waiting_for_continue:
                vlm_future = None
                self.waiting_for_continue = True
                print("\nPress Enter to ask another question...")

        cap.release()
        cv2.destroyAllWindows()

    def get_usb_video_devices(self):
        # TODO remove after integration to hailo apps
        video_devices = [f'/dev/{device}' for device in os.listdir('/dev') if device.startswith('video')]
        usb_video_devices = []
        for device in video_devices:
            try:
                udevadm_cmd = ["udevadm", "info", "--query=all", "--name=" + device]
                result = subprocess.run(udevadm_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                output = result.stdout
                if "ID_BUS=usb" in output and ":capture:" in output:
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

def get_rpi_camera():
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if cap.isOpened():
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            if width > 0:
                cap.release()
                return 0
        cap.release()
    except Exception as e:
        print(f"Error checking RPi camera: {e}")
    return None
    
if __name__ == "__main__":
    # TODO
    # parser = get_default_parser()
    # options_menu = parser.parse_args()
    # if options_menu.input is None:
    #     video_source = str(Path(RESOURCES_ROOT_PATH_DEFAULT) / RESOURCES_VIDEOS_DIR_NAME / BASIC_PIPELINES_VIDEO_EXAMPLE_NAME)
    # elif options_menu.input == USB_CAMERA:
    #     video_source = get_usb_video_devices()
    #     if video_source:
    #         video_source = video_source[0]
    # elif options_menu.input == RPI_NAME_I:
    #     video_source = get_rpi_camera()
    # if not video_source:
    #     print(f'Provided argument "--input" is set to {options_menu.input}, however no available cameras found. Please connect a camera or specifiy different input method.')
    #     exit(1)
    parser = argparse.ArgumentParser(description='VLM App')
    parser.add_argument("--camera", type=int, default=0, help='Camera ID (default: 0 for first USB camera)')
    video_source = ''
    # app = App(camera=video_source, cmaera_type=options_menu.input)
    app = App(camera=video_source, cmaera_type='')
    app.run()
    sys.exit(0)