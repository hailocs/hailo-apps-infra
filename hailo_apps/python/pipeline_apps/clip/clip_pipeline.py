# region imports
# Standard library imports
import os
import setproctitle
import signal

# Third-party imports
import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gst', '1.0')
from gi.repository import Gtk

# Local application-specific imports
from hailo_apps.python.core.gstreamer.gstreamer_app import GStreamerApp, app_callback_class, dummy_callback
from hailo_apps.python.core.common.core import get_default_parser, detect_hailo_arch, get_resource_path
from hailo_apps.python.pipeline_apps.clip.text_image_matcher import text_image_matcher
from hailo_apps.python.pipeline_apps.clip import gui
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    QUEUE, 
    SOURCE_PIPELINE, 
    INFERENCE_PIPELINE, 
    INFERENCE_PIPELINE_WRAPPER, 
    TRACKER_PIPELINE, 
    USER_CALLBACK_PIPELINE, 
    DISPLAY_PIPELINE, 
    CROPPER_PIPELINE
)
from hailo_apps.python.core.common.defines import (
    RESOURCES_SO_DIR_NAME, 
    RESOURCES_MODELS_DIR_NAME, 
    RESOURCES_VIDEOS_DIR_NAME,
    RESOURCES_JSON_DIR_NAME,
    BASIC_PIPELINES_VIDEO_EXAMPLE_NAME,
    CLIP_APP_TITLE,
    CLIP_VIDEO_NAME,
    CLIP_PIPELINE,
    CLIP_DETECTION_PIPELINE,
    CLIP_DETECTION_JSON_NAME,
    DETECTION_POSTPROCESS_SO_FILENAME,
    CLIP_POSTPROCESS_SO_FILENAME,
    CLIP_CROPPER_POSTPROCESS_SO_FILENAME,
)
# endregion

class GStreamerClipApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        setproctitle.setproctitle(CLIP_APP_TITLE)
        if parser == None:
            parser = get_default_parser()
        parser.add_argument("--detector", "-d", type=str, choices=["person", "face", "none"], default="none", help="Which detection pipeline to use.")
        parser.add_argument("--json-path", type=str, default=None, help="Path to JSON file to load and save embeddings. If not set, embeddings.json will be used.")
        parser.add_argument("--detection-threshold", type=float, default=0.5, help="Detection threshold.")
        parser.add_argument("--disable-runtime-prompts", action="store_true", help="When set, app will not support runtime prompts. Default is False.")
        super().__init__(parser, user_data)
        if self.options_menu.input is None:
            self.json_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example_embeddings.json') if self.options_menu.json_path is None else self.options_menu.json_path
        else:
            self.json_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embeddings.json') if self.options_menu.json_path is None else self.options_menu.json_path
        self.app_callback = app_callback
        self.detector = self.options_menu.detector
        self.text_image_matcher = text_image_matcher
        self.text_image_matcher.set_threshold(self.options_menu.detection_threshold)
        self.win = gui.AppWindow(self.options_menu.detection_threshold, self.options_menu.disable_runtime_prompts, self.text_image_matcher, self.json_file)
        self.detection_batch_size = 8
        self.clip_batch_size = 8

        if self.options_menu.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError('Could not auto-detect Hailo architecture. Please specify --arch manually.')
            self.arch = detected_arch
        else:
            self.arch = self.options_menu.arch

        if BASIC_PIPELINES_VIDEO_EXAMPLE_NAME in self.video_source:
            self.video_source = get_resource_path(pipeline_name=None, resource_type=RESOURCES_VIDEOS_DIR_NAME, model=BASIC_PIPELINES_VIDEO_EXAMPLE_NAME)

        self.hef_path_detection = get_resource_path(pipeline_name=CLIP_DETECTION_PIPELINE, resource_type=RESOURCES_MODELS_DIR_NAME)
        self.hef_path_clip = get_resource_path(pipeline_name=CLIP_PIPELINE, resource_type=RESOURCES_MODELS_DIR_NAME)

        self.detection_config_json_path = get_resource_path(pipeline_name=None, resource_type=RESOURCES_JSON_DIR_NAME, model=CLIP_DETECTION_JSON_NAME)

        self.post_process_so_detection = get_resource_path(pipeline_name=None, resource_type=RESOURCES_SO_DIR_NAME, model=DETECTION_POSTPROCESS_SO_FILENAME)
        self.post_process_so_clip = get_resource_path(pipeline_name=None, resource_type=RESOURCES_SO_DIR_NAME, model=CLIP_POSTPROCESS_SO_FILENAME)
        self.post_process_so_cropper = get_resource_path(pipeline_name=None, resource_type=RESOURCES_SO_DIR_NAME, model=CLIP_CROPPER_POSTPROCESS_SO_FILENAME)

        self.detection_post_process_function_name = 'yolov5s_personface'  # 'yolov5_personface_letterbox'
        self.clip_post_process_function_name = 'filter'
        if self.options_menu.detector == 'person':
            self.class_id = 1
            self.cropper_post_process_function_name = 'person_cropper'
        elif self.options_menu.detector == 'face':
            self.class_id = 2
            self.cropper_post_process_function_name = 'face_cropper'
        else: # fast_sam
            self.class_id = 0
            self.cropper_post_process_function_name = 'object_cropper'
    
        self.create_pipeline()

    def run(self):
        self.win.connect('destroy', self.on_destroy)
        self.win.show_all()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        # Gtk.main()
        super().run()
        
    def on_destroy(self, window):
        window.quit_button_clicked()

    def get_pipeline_string(self):
        source_pipeline = SOURCE_PIPELINE(self.video_source, self.video_width, self.video_height, frame_rate=self.frame_rate, sync=self.sync)

        detection_pipeline = INFERENCE_PIPELINE(
                hef_path=self.hef_path_detection,
                post_process_so=self.post_process_so_detection,
                post_function_name=self.detection_post_process_function_name,
                batch_size=self.detection_batch_size,
                config_json=self.detection_config_json_path,
                scheduler_priority=31,
                scheduler_timeout_ms=100,
                name='detection_inference'
        )

        detection_pipeline_wrapper = ''
        if self.options_menu.detector != 'none':
            detection_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(detection_pipeline)

        clip_pipeline = INFERENCE_PIPELINE(
                hef_path=self.hef_path_clip,
                post_process_so=self.post_process_so_clip,
                batch_size=self.clip_batch_size,
                scheduler_priority=16,
                scheduler_timeout_ms=1000,
                name='clip_inference'
        )
    
        tracker_pipeline = TRACKER_PIPELINE(class_id=self.class_id, keep_past_metadata=True)

        clip_cropper_pipeline = CROPPER_PIPELINE(
            inner_pipeline=clip_pipeline,
            so_path=self.post_process_so_cropper,
            function_name=self.cropper_post_process_function_name,
            name='clip_cropper'
        )

        # Clip pipeline with muxer integration (no cropper)
        clip_pipeline_wrapper = f'tee name=clip_t hailomuxer name=clip_hmux \
            clip_t. ! {QUEUE(name="clip_bypass_q", max_size_buffers=20)} ! clip_hmux.sink_0 \
            clip_t. ! {QUEUE(name="clip_muxer_queue")} ! videoscale n-threads=4 qos=false ! {clip_pipeline} ! clip_hmux.sink_1 \
            clip_hmux. ! {QUEUE(name="clip_hmux_queue")} '

        display_pipeline = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps)

        user_callback_pipeline = USER_CALLBACK_PIPELINE()

        if self.detector == 'none':
            return (
                f'{source_pipeline} ! '
                f'{clip_pipeline_wrapper} ! '
                f'{user_callback_pipeline} ! '
                f'{display_pipeline}'
            )
        else:
            return (
                f'{source_pipeline} ! '
                f'{detection_pipeline_wrapper} ! '
                f'{tracker_pipeline} ! '
                f'{clip_cropper_pipeline} ! '
                f'{user_callback_pipeline} ! '
                f'{display_pipeline}'
            )

if __name__ == "__main__":
    user_data = app_callback_class()
    app = GStreamerClipApp(user_data, dummy_callback)
    app.run()