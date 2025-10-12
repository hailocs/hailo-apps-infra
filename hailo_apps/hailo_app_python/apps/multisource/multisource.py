# region imports
# Standard library imports

# Third-party imports
import gi

gi.require_version("Gst", "1.0")
import cv2

# Local application-specific imports
import hailo
from gi.repository import Gst

from hailo_apps.hailo_app_python.apps.multisource.multisource_pipeline import GStreamerMultisourceApp
from hailo_apps.hailo_app_python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)

# Logger
from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class

hailo_logger = get_logger(__name__)
# endregion imports


# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()


# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------


# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    for detection in detections:
        track_id = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)[0].get_id()
        print(f'Unified callback, {roi.get_stream_id()}_{detection.get_label()}_{track_id}')
    return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    # Create an instance of the user app callback class
    hailo_logger.info("Starting Hailo Detection Multi-source App...")
    user_data = user_app_callback_class()
    app = GStreamerMultisourceApp(app_callback, user_data)
    app.run()
