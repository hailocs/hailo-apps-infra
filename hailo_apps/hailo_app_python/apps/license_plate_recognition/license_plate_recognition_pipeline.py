import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import logging
import setproctitle
import hailo

from hailo_apps.hailo_app_python.core.common.core import (
    get_default_parser,
    get_resource_path,
)
from hailo_apps.hailo_app_python.core.common.installation_utils import (
    detect_hailo_arch,
)
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import (
    GStreamerApp,
    app_callback_class,
    dummy_callback,
)
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    CROPPER_PIPELINE,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE,
    TRACKER_PIPELINE,
    QUEUE,  
)
from hailo_apps.hailo_app_python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)
from hailo_apps.hailo_app_python.core.common.defines import (
    # App / pipeline
    LPR_APP_TITLE,
    LPR_PIPELINE,

    # Models & resources
    LPR_VEHICLE_HEF_NAME,
    LPR_LICENSE_DET_HEF_NAME,
    LPR_OCR_MODEL_NAME,

    # Resource folders
    RESOURCES_SO_DIR_NAME,
    RESOURCES_MODELS_DIR_NAME,
    RESOURCES_VIDEOS_DIR_NAME,
    RESOURCES_JSON_DIR_NAME,

    # Video demo selection
    BASIC_PIPELINES_VIDEO_EXAMPLE_NAME,
    LPR_VIDEO_NAME,

    # Post-process (vehicle det)
    LPR_VEHICLE_POSTPROCESS_FUNCTION,
    LPR_VEHICLE_POSTPROCESS_SO_FILENAME,

    # Post-process (license plate det)
    LPR_LICENSE_DET_POSTPROCESS_SO_FILENAME,
    LPR_LICENSE_DET_POSTPROCESS_FUNCTION,

    # Post-process (OCR)
    LPR_OCR_POSTPROCESS_SO_FILENAME,
    LPR_OCR_POSTPROCESS_FUNCTION,

    # Croppers / overlays / sinks 
    LPR_OVERLAY_SO,
    LPR_CROPPERS_SO,                       
    LPR_OCRSINK_SO,                        
    LPR_QUALITY_ESTIMATION_FUNCTION_NAME,  
    LPR_VEHICLE_CROPPER_FUNCTION
)

hailo_logger = logging.getLogger(__name__)


class GStreamerLPRApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        # Use default CLI parser if none is supplied
        if parser is None:
            parser = get_default_parser()
        parser.add_argument(
            "--pipeline",
            default="simple",
            help="The pipeline variant: 'simple' or 'complex'",
        )
        super().__init__(parser, user_data)

        self.batch_size = 2
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45
        self.pipeline_type = self.options_menu.pipeline

        # Determine the architecture if not specified
        if self.options_menu.arch is None:
            detected_arch = detect_hailo_arch()
            hailo_logger.debug("Auto-detected Hailo arch: %s", detected_arch)
            if detected_arch is None:
                hailo_logger.error("Could not auto-detect Hailo architecture.")
                raise ValueError(
                    "Could not auto-detect Hailo architecture. Please specify --arch."
                )
            self.arch = detected_arch
            print(f"Auto-detected Hailo architecture: {self.arch}")
        else:
            self.arch = self.options_menu.arch
            hailo_logger.debug("Using user-specified arch: %s", self.arch)

        # Vehicle detection resources
        self.vehicle_hef_path = get_resource_path(
            LPR_PIPELINE, RESOURCES_MODELS_DIR_NAME, LPR_VEHICLE_HEF_NAME
        )
        self.vehicle_post_process_so = get_resource_path(
            LPR_PIPELINE, RESOURCES_SO_DIR_NAME, LPR_VEHICLE_POSTPROCESS_SO_FILENAME
        )
        self.vehicle_post_function_name = LPR_VEHICLE_POSTPROCESS_FUNCTION
        self.vehicle_json = get_resource_path(
            LPR_PIPELINE, RESOURCES_JSON_DIR_NAME, LPR_VEHICLE_HEF_NAME
        )
        self.vehicle_cropper_function = LPR_VEHICLE_CROPPER_FUNCTION 

        # License-plate detection resources
        self.license_det_hef_path = get_resource_path(
            LPR_PIPELINE, RESOURCES_MODELS_DIR_NAME, LPR_LICENSE_DET_HEF_NAME
        )
        self.license_det_post_process_so = get_resource_path(
            LPR_PIPELINE, RESOURCES_SO_DIR_NAME, LPR_LICENSE_DET_POSTPROCESS_SO_FILENAME
        )
        self.license_det_post_function_name = LPR_LICENSE_DET_POSTPROCESS_FUNCTION
        self.license_json = get_resource_path(
            LPR_PIPELINE, RESOURCES_JSON_DIR_NAME, LPR_LICENSE_DET_HEF_NAME
        )

        # OCR resources
        self.ocr_hef_path = get_resource_path(
            LPR_PIPELINE, RESOURCES_MODELS_DIR_NAME, LPR_OCR_MODEL_NAME
        )
        self.ocr_post_process_so = get_resource_path(
            LPR_PIPELINE, RESOURCES_SO_DIR_NAME, LPR_OCR_POSTPROCESS_SO_FILENAME
        )
        self.ocr_post_function_name = LPR_OCR_POSTPROCESS_FUNCTION

        # Overlay / croppers / sinks
        self.lpr_overlay_so = get_resource_path(
            LPR_PIPELINE, RESOURCES_SO_DIR_NAME, LPR_OVERLAY_SO
        )
        self.lpr_ocrsink_so = get_resource_path(
            LPR_PIPELINE, RESOURCES_SO_DIR_NAME, LPR_OCRSINK_SO
        )
        self.lpr_croppers_so = get_resource_path(
            LPR_PIPELINE, RESOURCES_SO_DIR_NAME, LPR_CROPPERS_SO
        )
        self.lpr_quality_est_function = LPR_QUALITY_ESTIMATION_FUNCTION_NAME

        # Video source auto-selection
        if BASIC_PIPELINES_VIDEO_EXAMPLE_NAME in self.video_source:
            self.video_source = get_resource_path(
                pipeline_name=LPR_PIPELINE,
                resource_type=RESOURCES_VIDEOS_DIR_NAME,
                model=LPR_VIDEO_NAME,
            )

        self.app_callback = app_callback

        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        # Set process title
        setproctitle.setproctitle(LPR_APP_TITLE)
        hailo_logger.debug("Process title set to %s", LPR_APP_TITLE)

        self.create_pipeline()
        hailo_logger.debug("Pipeline created")

    def get_pipeline_string(self):
        if self.pipeline_type == "simple":
            return self.get_pipeline_string_simple()
        return self.get_pipeline_string_complex()

    def get_pipeline_string_simple(self):
        # 1) Source
        source_pipeline = SOURCE_PIPELINE(
            video_source=self.video_source,
            video_width=self.video_width,
            video_height=self.video_height,
            frame_rate=self.frame_rate,
            sync=self.sync,
        )

        # 2) Vehicle detection
        vehicle_detection = INFERENCE_PIPELINE(
            hef_path=self.vehicle_hef_path,
            post_process_so=self.vehicle_post_process_so,
            post_function_name=self.vehicle_post_function_name,
            config_json=self.vehicle_json,
            additional_params=self.thresholds_str,
            name="vehicle_detection",
        )
        vehicle_detection_wrapper = INFERENCE_PIPELINE_WRAPPER(vehicle_detection)

        # 3) Tracking
        tracker_pipeline = TRACKER_PIPELINE(
            class_id=-1,               # Track all classes
            kalman_dist_thr=0.5,
            iou_thr=0.6,
            keep_tracked_frames=2,
            keep_lost_frames=2,
            keep_past_metadata=True,
            name="hailo_tracker",
        )

        # 4) Plate detection (inner pipe for cropper)
        plate_detection = INFERENCE_PIPELINE(
            hef_path=self.license_det_hef_path,
            post_process_so=self.license_det_post_process_so,
            post_function_name=self.license_det_post_function_name,
            config_json=self.license_json,
            name="plate_detection",
        )

        # 5) Vehicle cropper w/ plate detection
        vehicle_cropper = CROPPER_PIPELINE(
            inner_pipeline=plate_detection,
            so_path=self.lpr_croppers_so,
            function_name=self.vehicle_cropper_function,  
            internal_offset=True,
            name="vehicle_cropper",
        )

        # 6) OCR
        ocr_detection = INFERENCE_PIPELINE(
            hef_path=self.ocr_hef_path,
            post_process_so=self.ocr_post_process_so,
            post_function_name=self.ocr_post_function_name,
            name="ocr_detection",
        )

        # 7) Display
        display_pipeline = DISPLAY_PIPELINE(
            video_sink=self.video_sink,
            sync=self.sync,
            show_fps=self.show_fps,
        )

        # Full graph
        pipeline_string = (
            f"{source_pipeline} ! "
            f"{vehicle_detection_wrapper} ! "
            f"{tracker_pipeline} ! "
            f"tee name=context_tee "

            # Display branch with overlay
            f"context_tee. ! queue ! "
            f"videobox top=1 bottom=1 ! queue ! "
            f"hailooverlay line-thickness=3 font-thickness=1 qos=false ! "
            f"hailofilter use-gst-buffer=true "
            f"so-path={self.lpr_overlay_so} "
            f"qos=false ! "
            f"videoconvert ! fpsdisplaysink video-sink=ximagesink text-overlay=false "
            f"name=hailo_display sync=true "

            # Processing branch: crop -> OCR -> sink
            f"context_tee. ! {vehicle_cropper} ! "
            f"hailoaggregator name=agg2 "
            f"vehicle_cropper. ! queue ! agg2. "
            f"vehicle_cropper. ! queue ! {ocr_detection} ! queue ! agg2. "
            f"agg2. ! queue ! "
            f"identity name=identity_callback ! "
            f"hailofilter use-gst-buffer=true "
            f"so-path={self.lpr_ocrsink_so} "
            f"qos=false ! "
            f"fakesink sync=false async=false"
        )
        print(pipeline_string)
        return pipeline_string

    def get_pipeline_string_complex(self):
        # 1) Source
        source_pipeline = SOURCE_PIPELINE(
            video_source=self.video_source,
            video_width=self.video_width,
            video_height=self.video_height,
            frame_rate=self.frame_rate,
            sync=self.sync,
        )

        # 2) Vehicle detection (explicit stages)
        vehicle_detection = (
            f"{QUEUE('vehicle_pre_scale_q', max_size_buffers=30)} ! "
            f"videoscale name=vehicle_videoscale n-threads=2 qos=false ! "
            f"{QUEUE('vehicle_pre_convert_q', max_size_buffers=30)} ! "
            f"video/x-raw, pixel-aspect-ratio=1/1 ! "
            f"videoconvert name=vehicle_videoconvert n-threads=2 ! "
            f"{QUEUE('vehicle_pre_hailonet_q', max_size_buffers=30)} ! "
            f"hailonet name=vehicle_hailonet "
            f"hef-path={self.vehicle_hef_path} "
            f"vdevice-group-id=1 scheduling-algorithm=1 scheduler-threshold=1 scheduler-timeout-ms=100 "
            f"{self.thresholds_str} force-writable=true ! "
            f"{QUEUE('vehicle_post_hailonet_q', max_size_buffers=30)} ! "
            f"hailofilter name=vehicle_hailofilter "
            f"so-path={self.vehicle_post_process_so} "
            f"function-name={self.vehicle_post_function_name} "
            f"config-path={self.vehicle_json} qos=false ! "
            f"{QUEUE('vehicle_post_filter_q', max_size_buffers=30)} "
        )

        # 3) Tracker
        tracker_pipeline = (
            f"hailotracker name=hailo_tracker "
            f"keep-past-metadata=true kalman-dist-thr=0.5 iou-thr=0.6 "
            f"keep-tracked-frames=2 keep-lost-frames=2 ! "
            f"{QUEUE('tracker_post_q', max_size_buffers=30)} "
        )

        # 4) Plate detection (inner)
        plate_detection_inner = (
            f"hailonet name=plate_hailonet "
            f"hef-path={self.license_det_hef_path} "
            f"vdevice-group-id=1 scheduling-algorithm=1 scheduler-threshold=5 scheduler-timeout-ms=100 ! "
            f"{QUEUE('plate_post_hailonet_q', max_size_buffers=30)} ! "
            f"hailofilter name=plate_hailofilter "
            f"so-path={self.license_det_post_process_so} "
            f"config-path={self.license_json} "
            f"function-name={self.license_det_post_function_name} qos=false ! "
            f"{QUEUE('plate_post_filter_q', max_size_buffers=30)} "
        )

        # 5) OCR (inner)
        ocr_detection_inner = (
            f"hailonet name=ocr_hailonet "
            f"hef-path={self.ocr_hef_path} "
            f"vdevice-group-id=1 scheduling-algorithm=1 scheduler-threshold=1 scheduler-timeout-ms=100 ! "
            f"{QUEUE('ocr_post_hailonet_q', max_size_buffers=30)} ! "
            f"hailofilter name=ocr_hailofilter "
            f"so-path={self.ocr_post_process_so} qos=false ! "
            f"{QUEUE('ocr_post_filter_q', max_size_buffers=30)} "
        )

        # Full graph (explicit queues/aggregators)
        pipeline_string = (
            f"{source_pipeline} ! "
            f"{vehicle_detection} ! "
            f"{tracker_pipeline} ! "
            f"tee name=context_tee "

            # Display branch
            f"context_tee. ! {QUEUE('display_branch_q', max_size_buffers=30)} ! "
            f"videobox top=1 bottom=1 ! {QUEUE('display_videobox_q')} ! "
            f"hailooverlay line-thickness=3 font-thickness=1 qos=false ! "
            f"hailofilter use-gst-buffer=true "
            f"so-path={self.lpr_overlay_so} "
            f"qos=false ! "
            f"videoconvert ! fpsdisplaysink video-sink=ximagesink text-overlay=false "
            f"name=hailo_display sync=true "

            # Processing branch: crop -> plate det -> agg1
            f"context_tee. ! {QUEUE('processing_branch_q', max_size_buffers=30)} ! "
            f"hailocropper "
            f"so-path={self.lpr_croppers_so} "
            f"function-name={self.lpr_quality_est_function} "
            f"internal-offset=true drop-uncropped-buffers=true name=cropper1 "
            f"hailoaggregator name=agg1 "
            f"cropper1. ! {QUEUE('cropper1_bypass_q', max_size_buffers=50)} ! agg1.sink_0 "
            f"cropper1. ! {QUEUE('cropper1_process_q', max_size_buffers=30)} ! "
            f"{plate_detection_inner} ! agg1.sink_1 "
            f"agg1. ! {QUEUE('agg1_output_q', max_size_buffers=30)} ! "

            # Second crop -> OCR -> agg2
            f"hailocropper "
            f"so-path={self.lpr_croppers_so} "
            f"function-name={self.lpr_quality_est_function} "
            f"internal-offset=true drop-uncropped-buffers=true name=cropper2 "
            f"hailoaggregator name=agg2 "
            f"cropper2. ! {QUEUE('cropper2_bypass_q', max_size_buffers=50)} ! agg2.sink_0 "
            f"cropper2. ! {QUEUE('cropper2_process_q', max_size_buffers=30)} ! "
            f"{ocr_detection_inner} ! agg2.sink_1 "
            f"agg2. ! {QUEUE('agg2_output_q', max_size_buffers=30)} ! "

            # Final sink
            f"identity name=identity_callback ! "
            f"hailofilter use-gst-buffer=true "
            f"so-path={self.lpr_ocrsink_so} "
            f"qos=false ! "
            f"fakesink sync=false async=false"
        )
        print(pipeline_string)
        return pipeline_string


def main():
    hailo_logger.info("Starting Hailo LPR App...")
    user_data = app_callback_class()
    app_callback = dummy_callback
    app = GStreamerLPRApp(app_callback, user_data)
    app.run()


if __name__ == "__main__":
    main()
