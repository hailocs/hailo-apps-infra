# region imports
# Standard library imports
from pathlib import Path

import setproctitle

from hailo_apps.hailo_app_python.core.common.core import get_default_parser, get_resource_path
from hailo_apps.hailo_app_python.core.common.defines import (
OCR_APP_TITLE ,
OCR_PIPELINE,
OCR_DET_PIPELINE ,
OCR_DET_MODEL_NAME ,
OCR_DET_POSTPROCESS_FUNCTION ,
OCR_REC_PIPELINE,
OCR_REC_MODEL_NAME ,
OCR_REC_POSTPROCESS_FUNCTION ,
OCR_POSTPROCESS_SO_FILENAME ,
OCR_CROPPER_POSTPROCESS_FUNCTION ,
OCR_VIDEO_NAME,
RESOURCES_MODELS_DIR_NAME,
RESOURCES_SO_DIR_NAME,
RESOURCES_VIDEOS_DIR_NAME
)

# Logger
from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger

# Local application-specific imports
from hailo_apps.hailo_app_python.core.common.installation_utils import detect_hailo_arch
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import (
    GStreamerApp,
    app_callback_class,
    dummy_callback,
)
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import (
    DISPLAY_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    SOURCE_PIPELINE,
    TRACKER_PIPELINE,
    USER_CALLBACK_PIPELINE,
    CROPPER_PIPELINE,
)



hailo_logger = get_logger(__name__)

class GStreamerOCRApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        # use default CLI parser if none is supplied
        if parser is None:
            parser = get_default_parser()
        # initialise the base class; it parses CLI args and sets up video_source, etc.
        hailo_logger.info("Initializing GStreamer OCR App...")
        super().__init__(parser, user_data)

        # Hailo architecture (same flow used in other apps)
        if self.options_menu.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                hailo_logger.error("Could not auto-detect Hailo architecture.")
                raise ValueError(
                    "Could not auto-detect Hailo architecture. Please specify --arch manually."
                )
            self.arch = detected_arch
            hailo_logger.info("Auto-detected Hailo architecture: %s", self.arch)
        else:
            self.arch = self.options_menu.arch
            hailo_logger.debug("Using user-specified architecture: %s", self.arch)
        
        self.det_hef_path = str(
            get_resource_path(
                pipeline_name=OCR_DET_PIPELINE,
                resource_type=RESOURCES_MODELS_DIR_NAME,
            )
        )


        # store the callback so GStreamerApp can hook it up at run time
        self.app_callback = app_callback
        self.det_hef_path = str(
            get_resource_path(
                pipeline_name=OCR_DET_PIPELINE,
                resource_type=RESOURCES_MODELS_DIR_NAME,
            )
        )
        self.rec_hef_path = str(
            get_resource_path(
                pipeline_name=OCR_REC_PIPELINE,
                resource_type=RESOURCES_MODELS_DIR_NAME,
            )
        )

        self.det_post_function = OCR_DET_POSTPROCESS_FUNCTION
        self.rec_post_function = OCR_REC_POSTPROCESS_FUNCTION
        self.cropper_function = OCR_CROPPER_POSTPROCESS_FUNCTION

        self.post_process_so = str(
                get_resource_path(
                    pipeline_name=OCR_PIPELINE,
                    resource_type=RESOURCES_SO_DIR_NAME,
                    file_name=OCR_POSTPROCESS_SO_FILENAME,
                )
            )
        
        # override the video source and dimensions to match the detector input (960×544)
        if self.options_menu.input is None:
            self.video_source = get_resource_path(
                pipeline_name=OCR_REC_PIPELINE,
                resource_type=RESOURCES_VIDEOS_DIR_NAME,
                model=OCR_VIDEO_NAME,
            )
        self.video_width = 960
        self.video_height = 544

        setproctitle.setproctitle("Hailo OCR App")
        self.create_pipeline()

    def get_pipeline_string(self):
        # source stage
        source = SOURCE_PIPELINE(
            video_source=self.video_source,
            video_width=self.video_width,
            video_height=self.video_height,
            frame_rate=self.frame_rate,
            sync=self.sync,
        )
        # text detection stage; wrap it to preserve original frame size
        det = INFERENCE_PIPELINE(
            hef_path=self.det_hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.det_post_function,
        )
        det_wrapper = INFERENCE_PIPELINE_WRAPPER(det)

        # text recognition stage
        rec = INFERENCE_PIPELINE(
            hef_path=self.rec_hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.rec_post_function,
            name='recognition'   
        )

        # cropper stage: crops detections and feeds them into the recognition network
        cropper = CROPPER_PIPELINE(
            inner_pipeline=rec,
            so_path=self.post_process_so,
            function_name=self.cropper_function,
        )

        # user callback and display
        callback = USER_CALLBACK_PIPELINE()
        display = DISPLAY_PIPELINE(
            video_sink=self.video_sink,
            sync=self.sync,
            show_fps=self.show_fps,
        )

        return f"{source} ! {det_wrapper} ! {cropper} ! {callback} ! {display}"



def main():
    hailo_logger.info("Starting Hailo OCR App...")
    user_data = app_callback_class()
    app = GStreamerOCRApp(dummy_callback, user_data)
    app.run()


if __name__ == "__main__":
    hailo_logger.info("Executing __main__")
    main()