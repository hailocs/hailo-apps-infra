from pathlib import Path

# Base Defaults
HAILO8_ARCH = "hailo8"
HAILO8L_ARCH = "hailo8l"
AUTO_DETECT = "auto"
HAILO_TAPPAS = "hailo-tappas"
HAILO_TAPPAS_CORE = "hailo-tappas-core"
HAILO_TAPPAS_CORE_PYTHON = "hailo-tappas-core-python-binding"
HAILO_TAPPAS_CORE_PYTHON_NAMES = [HAILO_TAPPAS_CORE_PYTHON, "tappas-core-python-binding" , HAILO_TAPPAS_CORE]
HAILORT_PACKAGE = "hailort"
HAILO_FILE_EXTENSION = ".hef"
MODEL_ZOO_URL = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled"
RESOURCES_ROOT_PATH_DEFAULT = "/usr/local/hailo/resources" # Do Not Change!

# Core defaults
ARM_POSSIBLE_NAME = ["arm", "aarch64"]
X86_POSSIBLE_NAME = ["x86", "amd64", "x86_64"]
RPI_POSSIBLE_NAME = ["rpi", "raspberrypi", "pi"]
HAILO8_ARCH_CAPS = "HAILO8"
HAILO8L_ARCH_CAPS = "HAILO8L"
HAILO_FW_CONTROL_CMD = "hailortcli fw-control identify"
X86_NAME_I = "x86"
RPI_NAME_I = "rpi"
ARM_NAME_I = "arm"
LINUX_SYSTEM_NAME_I = "linux"
UNKNOWN_NAME_I = "unknown"
USB_CAMERA = "usb"
X86_LINUX_PLATFORM_TAG = "linux_x86_64"
ARM_LINUX_PLATFORM_TAG = "linux_aarch64"
CONFIG_DEFAULT_NAME = "config.yaml"
JSON_FILE_EXTENSION = ".json"

# CLI defaults
PYTHON_CMD = "python3"
PIP_CMD = "pip3"
VENV_CREATE_CMD = "python3 -m venv"

# Module names
HAILO_CORE_MODULE = "hailo_core"
HAILO_APPS_MODULE = "hailo_apps"
HAILO_COMMON_MODULE = "hailo_common"
HAILO_CONFIG_MODULE = "hailo_config"
HAILO_INSTALLATION_MODULE = "hailo_installation"
HAILO_GSTREAMER_MODULE = "hailo_gstreamer"
HAILO_PIPELINES_MODULE = "hailo_pipelines"
HAILO_MODULE_NAMES = [
    HAILO_COMMON_MODULE,
    HAILO_CONFIG_MODULE,
    HAILO_INSTALLATION_MODULE,
    HAILO_GSTREAMER_MODULE,
    HAILO_PIPELINES_MODULE,
]

# Base project paths
REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "hailo_apps_infra"
CORE_ROOT = PACKAGE_ROOT / HAILO_CORE_MODULE
APPS_ROOT = PACKAGE_ROOT / HAILO_APPS_MODULE
COMMON_ROOT = CORE_ROOT / HAILO_COMMON_MODULE
CONFIG_ROOT = CORE_ROOT / HAILO_CONFIG_MODULE
INSTALLATION_ROOT = CORE_ROOT / HAILO_INSTALLATION_MODULE
GSTREAMER_ROOT = APPS_ROOT / HAILO_GSTREAMER_MODULE
PIPELINES_ROOT = APPS_ROOT / HAILO_PIPELINES_MODULE

# Default config paths
DEFAULT_CONFIG_PATH = str(CONFIG_ROOT / "config.yaml")
DEFAULT_RESOURCES_CONFIG_PATH = str(CONFIG_ROOT / "resources_config.yaml")

# Symlink, dotenv, local resources defaults
DEFAULT_RESOURCES_SYMLINK_PATH = str(REPO_ROOT / "resources")
DEFAULT_DOTENV_PATH = str(REPO_ROOT / ".env")
DEFAULT_LOCAL_RESOURCES_PATH = str(REPO_ROOT / "local_resources")

# Supported config options
VALID_HAILORT_VERSION = [AUTO_DETECT, "4.20.0", "4.21.0", "4.22.0"]
VALID_TAPPAS_VERSION = [AUTO_DETECT, "3.30.0", "3.31.0", "3.32.0"]
VALID_MODEL_ZOO_VERSION = ["v2.13.0", "v2.14.0", "v2.15.0"]
VALID_HOST_ARCH = [AUTO_DETECT, "x86", "rpi", "arm"]
VALID_HAILO_ARCH = [AUTO_DETECT, "hailo8", "hailo8l"]
VALID_SERVER_URL = ["http://dev-public.hailo.ai/2025_01"]
VALID_TAPPAS_VARIANT = [AUTO_DETECT, HAILO_TAPPAS, HAILO_TAPPAS_CORE]

# Config key constants
HAILORT_VERSION_KEY = "hailort_version"
TAPPAS_VERSION_KEY = "tappas_version"
MODEL_ZOO_VERSION_KEY = "model_zoo_version"
HOST_ARCH_KEY = "host_arch"
HAILO_ARCH_KEY = "hailo_arch"
SERVER_URL_KEY = "server_url"
TAPPAS_VARIANT_KEY = "tappas_variant"
RESOURCES_PATH_KEY = "resources_path"
VIRTUAL_ENV_NAME_KEY = "virtual_env_name"
TAPPAS_POSTPROC_PATH_KEY = "tappas_postproc_path"
HAILO_APPS_INFRA_PATH_KEY = "hailo_apps_infra_path"

# Environment variable groups
DIC_CONFIG_VARIANTS = [
    HAILORT_VERSION_KEY,
    TAPPAS_VERSION_KEY,
    MODEL_ZOO_VERSION_KEY,
    HOST_ARCH_KEY,
    HAILO_ARCH_KEY,
    SERVER_URL_KEY,
    TAPPAS_VARIANT_KEY,
    RESOURCES_PATH_KEY,
    VIRTUAL_ENV_NAME_KEY,
    TAPPAS_POSTPROC_PATH_KEY,
]

# Default config values
HAILORT_VERSION_DEFAULT = AUTO_DETECT
TAPPAS_VERSION_DEFAULT = AUTO_DETECT
TAPPAS_VARIANT_DEFAULT = AUTO_DETECT
HOST_ARCH_DEFAULT = AUTO_DETECT
HAILO_ARCH_DEFAULT = AUTO_DETECT
MODEL_ZOO_VERSION_DEFAULT = "v2.14.0"
SERVER_URL_DEFAULT = "http://dev-public.hailo.ai/2025_01"
RESOURCES_PATH_DEFAULT = str(REPO_ROOT / "resources")
VIRTUAL_ENV_NAME_DEFAULT = "hailo_infra_venv"
STORAGE_PATH_DEFAULT = str(REPO_ROOT /"storage_deb_whl_dir")

# Resource groups for download_resources
RESOURCES_GROUP_DEFAULT = "default"
RESOURCES_GROUP_ALL = "all"
RESOURCES_GROUP_HAILO8 = "hailo8"
RESOURCES_GROUP_HAILO8L = "hailo8l"
RESOURCES_GROUP_RETRAIN = "retrain"

RESOURCES_GROUPS_MAP = [ RESOURCES_GROUP_DEFAULT,
                        RESOURCES_GROUP_ALL,
                        RESOURCES_GROUP_HAILO8,
                        RESOURCES_GROUP_HAILO8L,
                        RESOURCES_GROUP_RETRAIN]

# YAML config file keys
RESOURCES_CONFIG_DEFAULTS_KEY = "defaults"
RESOURCES_CONFIG_GROUPS_KEY = "models"
RESOURCES_CONFIG_VIDEOS_KEY = "videos"

# Resources directory structure
RESOURCES_MODELS_DIR_NAME = "models"
RESOURCES_VIDEOS_DIR_NAME = "videos"
RESOURCES_SO_DIR_NAME = "so"
RESOURCES_PHOTOS_DIR_NAME = "photos"
RESOURCES_GIF_DIR_NAME = "gifs"
RESOURCES_JSON_DIR_NAME = "json"
RESOURCE_STORAGE_DIR_NAME = "installation-storage"
RESOURCES_DIRS_MAP = [
    f"{RESOURCES_ROOT_PATH_DEFAULT}/{RESOURCES_MODELS_DIR_NAME}/{HAILO8_ARCH}",
    f"{RESOURCES_ROOT_PATH_DEFAULT}/{RESOURCES_MODELS_DIR_NAME}/{HAILO8L_ARCH}",
    f"{RESOURCES_ROOT_PATH_DEFAULT}/{RESOURCES_SO_DIR_NAME}",
    f"{RESOURCES_ROOT_PATH_DEFAULT}/{RESOURCES_PHOTOS_DIR_NAME}",
    f"{RESOURCES_ROOT_PATH_DEFAULT}{RESOURCES_GIF_DIR_NAME}",
    f"{RESOURCES_ROOT_PATH_DEFAULT}{RESOURCES_JSON_DIR_NAME}",
    f"{RESOURCES_ROOT_PATH_DEFAULT}/{RESOURCES_VIDEOS_DIR_NAME}",
    f"{RESOURCES_ROOT_PATH_DEFAULT}/{RESOURCE_STORAGE_DIR_NAME}",
]

# GStreamer defaults
GST_REQUIRED_VERSION = "1.0"

# Depth pipeline defaults
DEPTH_APP_TITLE = "Hailo Depth App"
DEPTH_PIPELINE = "depth"
DEPTH_POSTPROCESS_SO_FILENAME = "libdepth_postprocess.so"
DEPTH_POSTPROCESS_FUNCTION = "filter_scdepth"
DEPTH_MODEL_NAME = "scdepthv3"

# Simple detection pipeline defaults
SIMPLE_DETECTION_APP_TITLE = "Hailo Simple Detection App"
SIMPLE_DETECTION_PIPELINE = "simple_detection"
SIMPLE_DETECTION_VIDEO_NAME = "example_640.mp4"
SIMPLE_DETECTION_MODEL_NAME = "yolov6n"
SIMPLE_DETECTION_POSTPROCESS_SO_FILENAME = "libyolo_hailortpp_postprocess.so"
SIMPLE_DETECTION_POSTPROCESS_FUNCTION = "filter"

# Detection pipeline defaults
DETECTION_APP_TITLE = "Hailo Detection App"
DETECTION_PIPELINE = "detection"
DETECTION_MODEL_NAME_H8 = "yolov8m"
DETECTION_MODEL_NAME_H8L = "yolov8s"
DETECTION_POSTPROCESS_SO_FILENAME = "libyolo_hailortpp_postprocess.so"
DETECTION_POSTPROCESS_FUNCTION = "filter_letterbox"

# Instance segmentation pipeline defaults
INSTANCE_SEGMENTATION_APP_TITLE = "Hailo Instance Segmentation App"
INSTANCE_SEGMENTATION_PIPELINE = "instance_segmentation"
INSTANCE_SEGMENTATION_POSTPROCESS_SO_FILENAME = "libyolov5seg_postprocess.so"
INSTANCE_SEGMENTATION_POSTPROCESS_FUNCTION = "filter_letterbox"
INSTANCE_SEGMENTATION_MODEL_NAME_H8 = "yolov5m_seg"
INSTANCE_SEGMENTATION_MODEL_NAME_H8L = "yolov5n_seg"

# Pose estimation pipeline defaults
POSE_ESTIMATION_APP_TITLE = "Hailo Pose Estimation App"
POSE_ESTIMATION_PIPELINE = "pose_estimation"
POSE_ESTIMATION_POSTPROCESS_SO_FILENAME = "libyolov8pose_postprocess.so"
POSE_ESTIMATION_POSTPROCESS_FUNCTION = "filter_letterbox"
POSE_ESTIMATION_MODEL_NAME_H8 = "yolov8m_pose"
POSE_ESTIMATION_MODEL_NAME_H8L = "yolov8s_pose"

# Installation & subprocess defaults
PIP_SHOW_TIMEOUT = 5  # seconds
INSTALL_LOG = "env_setup.log"

# Testing defaults
TEST_RUN_TIME = 10  # seconds
TERM_TIMEOUT = 5    # seconds

# USB device discovery
UDEV_CMD = "udevadm"

# Miscellaneous
EPSILON = 1e-6

# Compile_cpp defaults
MODE_RELEASE = "release"
MODE_DEBUG = "debug"
MODE_CLEAN = "clean"

# Download resources defaults
DEFAULT_VIDEO_FORMAT_SUFFIX = ".mp4"

HAILO_RGB_VIDEO_FORMAT = "RGB"
HAILO_BGR_VIDEO_FORMAT = "BGR"
HAILO_YUYV_VIDEO_FORMAT = "YUYV"
HAILO_NV12_VIDEO_FORMAT = "NV12"

# Video examples
BASIC_PIPELINES_VIDEO_EXAMPLE_NAME = "example.mp4"
BASIC_PIPELINES_VIDEO_EXAMPLE_640_NAME = "example_640.mp4"

# Gstreamer pipeline defaults
GST_VIDEO_SINK = "autovideosink"
