s# region imports
# Standard library imports

# Third-party imports
import gi

gi.require_version("Gst", "1.0")
import cv2

# Local application-specific imports
import hailo
from gi.repository import Gst

from hailo_apps.hailo_app_python.apps.license_plate_recognition.license_plate_recognition_pipeline import GStreamerLPRApp
from hailo_apps.hailo_app_python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)

# Logger
from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class

hailo_logger = get_logger(__name__)

# User-defined class to store OCR results  
class user_app_callback_class(app_callback_class):  
    def __init__(self):  
        super().__init__()  
        self.ocr_results = []  # Store OCR text results  
        self.output_file = "ocr_results.txt"  # File to write results  
  
    def write_ocr_text(self, text, vehicle_id=None, confidence=None):  
        """Write OCR text to file with optional metadata"""  
        with open(self.output_file, "a", encoding="utf-8") as f:  
            timestamp = f"Frame {self.get_count()}"  
            if vehicle_id is not None:  
                f.write(f"{timestamp} - Vehicle ID {vehicle_id}: {text}")  
            else:  
                f.write(f"{timestamp}: {text}")  
            if confidence is not None:  
                f.write(f" (Confidence: {confidence:.2f})")  
            f.write("\n")  
  
# OCR callback function  
def app_callback(pad, info, user_data):  
    user_data.increment()  # Count frames  
    buffer = info.get_buffer()  
    if buffer is None:  
        return Gst.PadProbeReturn.OK  
  
    # Get the ROI from buffer  
    roi = hailo.get_roi_from_buffer(buffer)  
      
    # Extract vehicle detections first  
    vehicle_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)  
      
    for vehicle_detection in vehicle_detections:  
        if vehicle_detection.get_label() == "vehicle":  
            # Get vehicle track ID  
            vehicle_id = None  
            track = vehicle_detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)  
            if len(track) > 0:  
                vehicle_id = track[0].get_id()  
              
            # Look for license plate detections within this vehicle  
            license_plates = vehicle_detection.get_objects_typed(hailo.HAILO_DETECTION)  
            for plate_detection in license_plates:  
                if plate_detection.get_label() == "license_plate":  
                    # Look for OCR text classifications  
                    ocr_classifications = plate_detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)  
                    for ocr_result in ocr_classifications:  
                        if ocr_result.get_label():  # OCR text result  
                            ocr_text = ocr_result.get_label()  
                            confidence = ocr_result.get_confidence()  
                              
                            # Write OCR text to file  
                            user_data.write_ocr_text(ocr_text, vehicle_id, confidence)  
                              
                            # Also print to console  
                            print(f"OCR Result: '{ocr_text}' (Vehicle ID: {vehicle_id}, Confidence: {confidence:.2f})")  
  
    return Gst.PadProbeReturn.OK

def main():  
    hailo_logger.info("Starting Hailo LPR App...")  
    user_data = user_app_callback_class()  # Use custom callback class  
    app = GStreamerLPRApp(app_callback, user_data)  # Use OCR callback  
    app.run()

