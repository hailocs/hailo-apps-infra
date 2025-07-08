# region imports
# Standard library imports
import setproctitle
import json
import os

# Third-party imports
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Local application-specific imports
import hailo
from hailo_apps.hailo_app_python.core.common.core import get_default_parser, get_resource_path, get_resource_path
from hailo_apps.hailo_app_python.core.common.db_handler import DatabaseHandler, Record
from hailo_apps.hailo_app_python.core.common.installation_utils import detect_hailo_arch
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.common.defines import MULTI_SOURCE_DIR_NAME, MULTI_SOURCE_DATABASE_DIR_NAME, MULTI_SOURCE_PARAMS_JSON_NAME, RESOURCES_JSON_DIR_NAME, MULTISOURCE_APP_TITLE, SIMPLE_DETECTION_PIPELINE, RESOURCES_MODELS_DIR_NAME, RESOURCES_SO_DIR_NAME, DETECTION_POSTPROCESS_SO_FILENAME, DETECTION_POSTPROCESS_FUNCTION, TAPPAS_POSTPROC_PATH_KEY
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import get_source_type, USER_CALLBACK_PIPELINE, TRACKER_PIPELINE, QUEUE, SOURCE_PIPELINE, INFERENCE_PIPELINE, DISPLAY_PIPELINE
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import GStreamerApp, app_callback_class
# endregion imports

# User Gstreamer Application: This class inherits from the common.GStreamerApp class
class GStreamerMultisourceApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        
        if parser == None:
            parser = get_default_parser()
        parser.add_argument("--sources", default='', help="The list of sources to use for the multisource pipeline, separated with comma e.g., /dev/video0,/dev/video1")

        super().__init__(parser, user_data)  # Call the parent class constructor

        # Determine the architecture if not specified
        if self.options_menu.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError('Could not auto-detect Hailo architecture. Please specify --arch manually.')
            self.arch = detected_arch
        else:
            self.arch = self.options_menu.arch

        setproctitle.setproctitle(MULTISOURCE_APP_TITLE)  # Set the process title

        self.hef_path = get_resource_path(SIMPLE_DETECTION_PIPELINE, RESOURCES_MODELS_DIR_NAME)
        self.post_process_so = get_resource_path(SIMPLE_DETECTION_PIPELINE, RESOURCES_SO_DIR_NAME, DETECTION_POSTPROCESS_SO_FILENAME)
        self.post_function_name = DETECTION_POSTPROCESS_FUNCTION
        self.video_sources_types = [(video_source, get_source_type(video_source)) for video_source in (self.options_menu.sources.split(',') if self.options_menu.sources else [self.video_source, self.video_source])]  # Default to 2 sources if none specified
        self.num_sources = len(self.video_sources_types)  
        self.algo_params = json.load(open(get_resource_path(pipeline_name=None, resource_type=RESOURCES_JSON_DIR_NAME, model=MULTI_SOURCE_PARAMS_JSON_NAME), "r+"))
        self.video_height = 640
        self.video_width = 640
        # self.frame_rate = 15

        self.app_callback = app_callback
        self.generate_callbacks()        
        self.create_pipeline()
        self.connect_src_callbacks()

        # Initialize the database and table
        self.db_handler = DatabaseHandler(db_name='cross_tracked.db', 
                                          table_name='cross_tracked', 
                                          schema=Record, 
                                          threshold=self.algo_params['lance_db_vector_search_classificaiton_confidence_threshold'],
                                          database_dir=get_resource_path(pipeline_name=None, resource_type=MULTI_SOURCE_DIR_NAME, model=MULTI_SOURCE_DATABASE_DIR_NAME),
                                          samples_dir=None)

    def get_pipeline_string(self):
        sources_string = ''
        router_string = ''

        tappas_post_process_dir = os.environ.get(TAPPAS_POSTPROC_PATH_KEY, '')
        tappas_post_process_dir = '/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes'  # TODO
        set_stream_id_so = os.path.join(tappas_post_process_dir, 'libstream_id_tool.so')
        for id in range(self.num_sources):
            sources_string += SOURCE_PIPELINE(video_source=self.video_sources_types[id][0], 
                                              video_width=self.video_width, video_height=self.video_height, 
                                              frame_rate=self.frame_rate, sync=self.sync, name=f"source_{id}", no_webcam_compression=True)
            sources_string += f"! hailofilter name=set_src_{id} so-path={set_stream_id_so} config-path=f'src_{id}' "
            sources_string += f"! robin.sink_{id} "
            router_string += f"router.src_{id} ! {USER_CALLBACK_PIPELINE(name=f'src_{id}_callback')} ! {QUEUE(name=f'callback_q_{id}')} ! {DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps, name=f'hailo_display_{id}')} "

        self.thresholds_str = (
            f"nms-score-threshold={self.algo_params['nms_score_threshold']} "
            f"nms-iou-threshold={self.algo_params['nms_iou_threshold']} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        # Create the detection pipeline
        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            batch_size=self.batch_size,
            additional_params=self.thresholds_str)

        # inference_string = f"hailoroundrobin mode=1 name=robin ! {detection_pipeline} ! hailostreamrouter name=router "
        inference_string = f"hailoroundrobin mode=1 name=robin ! {detection_pipeline} ! {TRACKER_PIPELINE(class_id=-1)} ! {USER_CALLBACK_PIPELINE()} ! {QUEUE(name='call_q')} ! hailostreamrouter name=router "
        for id in range(self.num_sources):
            inference_string += f"src_{id}::input-streams=\"<sink_{id}>\" "

        pipeline_string = sources_string + inference_string + router_string

        # print(pipeline_string)
        return pipeline_string
    
    def generate_callbacks(self):
        # Dynamically define callback functions per sources
        for id in range(self.num_sources):
            def callback_function(pad, info, user_data, id=id):
                buffer = info.get_buffer()
                if buffer is None:
                    return Gst.PadProbeReturn.OK
                roi = hailo.get_roi_from_buffer(buffer)
                detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
                string_to_print = ''
                track_id = -1
                for detection in detections:
                    if detection.get_label() == "person":
                        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
                        if len(track) == 1:
                            track_id = track[0].get_id()
                        string_to_print += (f"Callback source: {id}, person track id: {track_id}\n")
                print(string_to_print)
                return Gst.PadProbeReturn.OK

            # Attach the callback function to the instance
            setattr(self, f'src_{id}_callback', callback_function)
    
    def connect_src_callbacks(self):
        for id in range(self.num_sources):
            identity = self.pipeline.get_by_name(f'src_{id}_callback')
            identity_pad = identity.get_static_pad(f'src')
            callback_function = getattr(self, f'src_{id}_callback', None)
            identity_pad.add_probe(Gst.PadProbeType.BUFFER, callback_function, self.user_data)

def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    print(roi.get_stream_id())

    # Parse the detections
    detection_count = 0
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        if label == "person":
            # Get track ID
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()
            string_to_print += (f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n")
            detection_count += 1

    # print(string_to_print)
    return Gst.PadProbeReturn.OK

def main():
    # Create an instance of the user app callback class
    user_data = app_callback_class()
    app = GStreamerMultisourceApp(app_callback, user_data)
    app.run()
    
if __name__ == "__main__":
    print("Starting Hailo Multisource App...")
    main()