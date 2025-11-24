"""HEF file utilities for extracting model information."""

from pathlib import Path
from typing import Tuple, Optional

from .hailo_logger import get_logger
from .core import get_resource_path
from .defines import (
    RESOURCES_JSON_DIR_NAME,
    VISDRONE_CONFIG_JSON_NAME,
    HAILO_4_CLASSES_CONFIG_JSON_NAME,
)

hailo_logger = get_logger(__name__)

try:
    from hailo_platform import HEF
except ImportError:
    HEF = None
    hailo_logger.warning("hailo_platform not available. HEF utilities will not work.")


def get_hef_input_size(hef_path: str) -> Tuple[int, int]:
    """
    Get the input resolution (width, height) from a HEF file.

    Args:
        hef_path: Path to the HEF file

    Returns:
        tuple: (width, height) of the input resolution

    Raises:
        ImportError: If hailo_platform is not available
        FileNotFoundError: If the HEF file does not exist
        ValueError: If the HEF file has no input streams or cannot be parsed
    """
    if HEF is None:
        raise ImportError("hailo_platform is not available. Cannot parse HEF files.")

    hef_path_obj = Path(hef_path)
    if not hef_path_obj.exists():
        raise FileNotFoundError(f"HEF file not found: {hef_path}")

    try:
        # Load the HEF file
        hef = HEF(str(hef_path))

        # Get input vstream infos directly from HEF
        input_vstream_infos = hef.get_input_vstream_infos()

        if not input_vstream_infos:
            raise ValueError(f"No input streams found in HEF file: {hef_path}")

        # Get the first input stream's shape
        input_vstream_info = input_vstream_infos[0]
        shape = input_vstream_info.shape

        # Shape is typically [batch, height, width, channels] or [batch, channels, height, width]
        # Try to determine the format by checking the shape length and values
        if len(shape) == 4:
            # Common formats:
            # NHWC: [batch, height, width, channels]
            # NCHW: [batch, channels, height, width]
            # Assume NHWC if height and width are the larger dimensions
            if shape[1] > shape[3] and shape[2] > shape[3]:
                # Likely NHWC format
                height = shape[1]
                width = shape[2]
            elif shape[2] > shape[1] and shape[3] > shape[1]:
                # Likely NCHW format
                height = shape[2]
                width = shape[3]
            else:
                # Default: assume NHWC
                height = shape[1]
                width = shape[2]
        elif len(shape) == 3:
            # [height, width, channels] or [channels, height, width]
            if shape[0] > shape[2]:
                height = shape[0]
                width = shape[1]
            else:
                height = shape[1]
                width = shape[2]
        else:
            raise ValueError(
                f"Unexpected input shape format: {shape}. "
                f"Expected 3 or 4 dimensions, got {len(shape)}"
            )

        hailo_logger.debug(
            f"HEF input resolution: {width}x{height} (shape: {shape})"
        )

        return (width, height)

    except Exception as e:
        if isinstance(e, (ValueError, FileNotFoundError, ImportError)):
            raise
        raise ValueError(f"Failed to parse HEF file {hef_path}: {str(e)}") from e


def get_hef_input_shape(hef_path: str) -> Tuple[int, ...]:
    """
    Get the full input shape from a HEF file.

    Args:
        hef_path: Path to the HEF file

    Returns:
        tuple: The input shape (e.g., (1, 480, 640, 3) or (1, 3, 480, 640))

    Raises:
        ImportError: If hailo_platform is not available
        FileNotFoundError: If the HEF file does not exist
        ValueError: If the HEF file has no input streams or cannot be parsed
    """
    if HEF is None:
        raise ImportError("hailo_platform is not available. Cannot parse HEF files.")

    hef_path_obj = Path(hef_path)
    if not hef_path_obj.exists():
        raise FileNotFoundError(f"HEF file not found: {hef_path}")

    try:
        # Load the HEF file
        hef = HEF(str(hef_path))

        # Get input vstream infos directly from HEF
        input_vstream_infos = hef.get_input_vstream_infos()

        if not input_vstream_infos:
            raise ValueError(f"No input streams found in HEF file: {hef_path}")

        # Get the first input stream's shape
        input_vstream_info = input_vstream_infos[0]
        shape = tuple(input_vstream_info.shape)

        hailo_logger.debug(f"HEF input shape: {shape}")

        return shape

    except Exception as e:
        if isinstance(e, (ValueError, FileNotFoundError, ImportError)):
            raise
        raise ValueError(f"Failed to parse HEF file {hef_path}: {str(e)}") from e


def get_json_config_path_from_hef_name(hef_path: str) -> Optional[str]:
    """Check HEF name for specific models and return the path to the relevant JSON configuration file."""
    hef_name = Path(hef_path).name.lower()
    json_path = None
    if "visdrone" in hef_name:
        json_path = get_resource_path(pipeline_name=None,
                                      resource_type=RESOURCES_JSON_DIR_NAME,
                                      model=VISDRONE_CONFIG_JSON_NAME)
        hailo_logger.info(f"Detected 'visdrone' in HEF name, using config: {json_path}")
    elif "4_classes" in hef_name:
        json_path = get_resource_path(pipeline_name=None,
                                      resource_type=RESOURCES_JSON_DIR_NAME,
                                      model=HAILO_4_CLASSES_CONFIG_JSON_NAME)
        hailo_logger.info(f"Detected '4_classes' in HEF name, using config: {json_path}")
    return str(json_path) if json_path else None

