# region imports
# Standard library imports
import os
import setproctitle

# Third-party imports
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Local application-specific imports
import hailo
from hailo_apps.hailo_app_python.core.common.installation_utils import detect_hailo_arch
from hailo_apps.hailo_app_python.core.common.core import get_default_parser, get_resource_path
from hailo_apps.hailo_app_python.core.common.defines import (
    HAILO_ARCH_KEY,
    RESOURCES_MODELS_DIR_NAME,
    TILING_APP_TITLE,
    TILING_MODEL_NAME, 
    TILING_POSTPROCESS_SO_FILENAME, 
    TILING_POSTPROCESS_FUNCTION,
    RESOURCES_SO_DIR_NAME,
)
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import SOURCE_PIPELINE, INFERENCE_PIPELINE, USER_CALLBACK_PIPELINE, DISPLAY_PIPELINE, TILE_CROPPER_PIPELINE
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import GStreamerApp, app_callback_class, dummy_callback
# endregion imports


# Logger
from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)

# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------

# This class inherits from the hailo_rpi_common.GStreamerApp class
class GStreamerTilingApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        if parser == None:
            parser = get_default_parser()
        parser.add_argument("--tiles_along_x_axis", default=4, help="Set number of tiles along x axis (columns). Default is 4")
        parser.add_argument("--tiles_along_y_axis", default=3, help="Set number of tiles along y axis (rows). Default is 3")
        parser.add_argument("--overlap_x_axis", default=0.1, help="Set overlap in percentage between tiles along x axis (columns). Default is 0.1")
        parser.add_argument("--overlap_y_axis", default=0.08, help="Set overlap in percentage between tiles along y axis (rows). Default is 0.08")
        parser.add_argument("--iou_threshold", default=0.3, help="Set iou threshold for NMS. Default is 0.3")
        parser.add_argument("--border_threshold", default=0.1, help="Set border threshold to Remove tile's exceeded objects. Relevant only for multi scaling. Default is 0.1")
        parser.add_argument("--single_scaling", action="store_true", help="Whether use single scaling or multi scaling. Default is multi scaling.")
        parser.add_argument("--scale_level", default=2, help="set scales (layers of tiles) in addition to the main layer [1,2,3] 1: {(1 X 1)} 2: {(1 X 1), (2 X 2)} 3: {(1 X 1), (2 X 2), (3 X 3)}. Default is 2. For singlescaling must be 0.")

        hailo_logger.info("Initializing GStreamer Tiling App...")
        
        # Call the parent class constructor
        super().__init__(parser, user_data)

        hailo_logger.debug(
            "Parent GStreamerApp initialized | arch=%s | input=%s | fps=%s | sync=%s | show_fps=%s",
            getattr(self.options_menu, "arch", None),
            getattr(self, "video_source", None),
            getattr(self, "frame_rate", None),
            getattr(self, "sync", None),
            getattr(self, "show_fps", None),
        )
        
        if self.options_menu.single_scaling:
            self.options_menu.scale_level = 0
            self.options_menu.border_threshold = 0

        # Determine the architecture if not specified
        if self.options_menu.arch is None:    
            arch = os.getenv(HAILO_ARCH_KEY, detect_hailo_arch())
            if not arch:
                hailo_logger.error("Could not detect Hailo architecture.")
                raise ValueError(
                    "Could not auto-detect Hailo architecture. Please specify --arch manually."
                )
            self.arch = arch
            hailo_logger.debug(f"Auto-detected Hailo architecture: {self.arch}")
        else:
            self.arch = self.options_menu.arch
            hailo_logger.debug("Using user-specified arch: %s", self.arch)

        

        if self.options_menu.hef_path is not None:
            self.hef_path = self.options_menu.hef_path
        else:
            self.hef_path = get_resource_path(pipeline_name=None, resource_type=RESOURCES_MODELS_DIR_NAME, model=TILING_MODEL_NAME)

        self.post_process_so = get_resource_path(pipeline_name=None, resource_type=RESOURCES_SO_DIR_NAME, model=TILING_POSTPROCESS_SO_FILENAME)
        self.post_function = TILING_POSTPROCESS_FUNCTION

        self.app_callback = app_callback
        setproctitle.setproctitle(TILING_APP_TITLE)
        self.create_pipeline()

    def get_pipeline_string(self):
        source_pipeline = SOURCE_PIPELINE(
            video_source=self.video_source,
            video_width=self.video_width, video_height=self.video_height,
            frame_rate=self.frame_rate, sync=self.sync,
            no_webcam_compression=True)
        
        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function,
            batch_size=self.batch_size)
        
        tile_cropper_pipeline = TILE_CROPPER_PIPELINE(
            detection_pipeline,
            name='tile_cropper_wrapper',
            internal_offset=True,
            scale_level=self.options_menu.scale_level,
            tiling_mode=0 if self.options_menu.single_scaling else 1,
            tiles_along_x_axis=self.options_menu.tiles_along_x_axis,
            tiles_along_y_axis=self.options_menu.tiles_along_y_axis,
            overlap_x_axis=self.options_menu.overlap_x_axis,
            overlap_y_axis=self.options_menu.overlap_y_axis,
            iou_threshold=self.options_menu.iou_threshold,
            border_threshold=0 if self.options_menu.single_scaling else self.options_menu.border_threshold
        )

        user_callback_pipeline = USER_CALLBACK_PIPELINE()

        display_pipeline = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps)

        pipeline_string = (
            f'{source_pipeline} ! '
            f'{tile_cropper_pipeline} ! '
            f'{user_callback_pipeline} ! '
            f'{display_pipeline}'
        )

        print(pipeline_string)
        return pipeline_string
    


def main():
    # Create an instance of the user app callback class
    user_data = app_callback_class()
    app_callback = dummy_callback
    app = GStreamerTilingApp(app_callback, user_data)
    app.run()

if __name__ == "__main__":
    print("Starting Hailo Tiling App...")
    main()
