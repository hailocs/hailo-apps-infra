# Standard library imports
import setproctitle
from pathlib import Path
from typing import Optional, Any

# Third-party imports
import gi
gi.require_version('Gst', '1.0')

# Local application-specific imports
from hailo_apps.python.core.common.core import get_default_parser
from hailo_apps.python.core.common.defines import TILING_APP_TITLE
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (SOURCE_PIPELINE, INFERENCE_PIPELINE,
                                                                       USER_CALLBACK_PIPELINE, DISPLAY_PIPELINE,
                                                                       TILE_CROPPER_PIPELINE)
from hailo_apps.python.core.gstreamer.gstreamer_app import GStreamerApp, app_callback_class, dummy_callback
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.pipeline_apps.tiling.configuration import TilingConfiguration

hailo_logger = get_logger(__name__)

# -----------------------------------------------------------------------------------------------
# Main Tiling Application Class
# -----------------------------------------------------------------------------------------------
class GStreamerTilingApp(GStreamerApp):
    def __init__(self, app_callback: Any, user_data: Any, parser: Optional[Any] = None) -> None:
        if parser is None:
            parser = get_default_parser()

        # Add tiling-specific arguments
        self._add_tiling_arguments(parser)

        # Call the parent class constructor
        super().__init__(parser, user_data)

        # Initialize tiling configuration
        self.config = TilingConfiguration(
            self.options_menu,
            self.video_width,
            self.video_height,
            self.arch
        )

        self.app_callback = app_callback
        setproctitle.setproctitle(TILING_APP_TITLE)

        # Print configuration summary
        self._print_configuration()

        self.create_pipeline()

    def _add_tiling_arguments(self, parser: Any) -> None:
        """Add tiling-specific command line arguments."""
        # Tiling options (auto mode by default, manual if tiles-x/y specified)
        parser.add_argument("--tiles-x", type=int, default=None,
                          help="Number of tiles horizontally (triggers manual mode)")
        parser.add_argument("--tiles-y", type=int, default=None,
                          help="Number of tiles vertically (triggers manual mode)")
        parser.add_argument("--min-overlap", type=float, default=0.1,
                          help="Minimum overlap ratio (0.0-0.5). Default: 0.1 (10%% of tile size). "
                               "Should be ≥ (smallest_object_size / model_input_dimension)")

        # Multi-scale options
        parser.add_argument("--multi-scale", action="store_true",
                          help="Enable multi-scale tiling with predefined grids")
        parser.add_argument("--scale-levels", type=int, default=1, choices=[1, 2, 3],
                          help="Scale levels for multi-scale mode: 1={1x1}, 2={1x1+2x2}, 3={1x1+2x2+3x3}. Default: 1")

        # Detection options
        parser.add_argument("--iou-threshold", type=float, default=0.3,
                          help="NMS IOU threshold (default: 0.3)")
        parser.add_argument("--border-threshold", type=float, default=0.15,
                          help="Border threshold for multi-scale mode (default: 0.15)")

    def _print_configuration(self) -> None:
        """
        Print a user-friendly configuration summary to the console.
        """
        print("\n" + "="*70)
        print("TILING CONFIGURATION")
        print("="*70)

        # Input information
        print(f"Input Resolution:     {self.video_width}x{self.video_height}")
        print(
            f"Model:                {Path(self.config.hef_path).name} "
            f"({self.config.model_type.upper()}, {self.config.model_input_width}x{self.config.model_input_height})"
        )

        # Tiling mode and configuration (always show custom tiles)
        print(f"\nTiling Mode:          {self.config.tiling_mode.upper()}")
        print(f"Custom Tile Grid:     {self.config.tiles_x}x{self.config.tiles_y} = {self.config.tiles_x * self.config.tiles_y} tiles")

        if self.config.used_larger_tiles:
            print(f"Tile Size:            {int(self.config.tile_size_x)}x{int(self.config.tile_size_y)} pixels (enlarged to meet min overlap)")
        else:
            print(f"Tile Size:            {self.config.model_input_width}x{self.config.model_input_height} pixels")
        # Calculate overlap in pixels using actual tile sizes
        overlap_pixels_x = int(self.config.overlap_x * self.config.tile_size_x)
        overlap_pixels_y = int(self.config.overlap_y * self.config.tile_size_y)
        print(f"Overlap:              X: {self.config.overlap_x*100:.1f}% (~{overlap_pixels_x}px), Y: {self.config.overlap_y*100:.1f}% (~{overlap_pixels_y}px)")

        # Multi-scale additional information
        if self.config.use_multi_scale:
            print(f"\nMulti-Scale:          ENABLED (scale-level={self.config.scale_level})")
            if self.config.scale_level == 1:
                print("  Additional Grids:   1x1 = 1 tile")
                predefined = 1
            elif self.config.scale_level == 2:
                print("  Additional Grids:   1x1 + 2x2 = 5 tiles")
                predefined = 5
            else:  # scale_level == 3
                print("  Additional Grids:   1x1 + 2x2 + 3x3 = 14 tiles")
                predefined = 14
            custom = self.config.tiles_x * self.config.tiles_y
            print(f"  Total Tiles:        {custom} (custom) + {predefined} (predefined) = {self.config.batch_size}")
        else:
            print("\nMulti-Scale:          DISABLED")
            print(f"  Total Tiles:        {self.config.batch_size}")

        # Detection parameters
        print("\nDetection Parameters:")
        print(f"  Batch Size:         {self.config.batch_size}")
        print(f"  IOU Threshold:      {self.config.iou_threshold}")
        if self.config.use_multi_scale:
            print(f"  Border Threshold:   {self.config.border_threshold}")

        # Overlap information
        # Use average for min overlap pixels display
        avg_model_size = (self.config.model_input_width + self.config.model_input_height) / 2
        min_overlap_pixels = int(self.config.min_overlap * avg_model_size)

        if self.config.used_larger_tiles:
            print(
                f"\nNote:                 Tile sizes enlarged to {int(self.config.tile_size_x)}x{int(self.config.tile_size_y)} "
                "to meet minimum overlap requirement"
            )

        if overlap_pixels_x < min_overlap_pixels or overlap_pixels_y < min_overlap_pixels:
            print(f"  ⚠️  Warning:         Overlap below minimum ({min_overlap_pixels}px)")
        elif overlap_pixels_x < 50 or overlap_pixels_y < 50:
            print("  ⚠️  Warning:         Very small overlap may miss objects on boundaries")

        print("="*70 + "\n")

    def get_pipeline_string(self) -> str:
        """
        Build the GStreamer pipeline string with configured tiling parameters.

        Returns:
            str: Complete GStreamer pipeline string
        """
        source_pipeline = SOURCE_PIPELINE(
            video_source=self.config.video_source,
            video_width=self.video_width,
            video_height=self.video_height,
            frame_rate=self.frame_rate,
            sync=self.sync,
        )

        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.config.hef_path,
            post_process_so=self.config.post_process_so,
            post_function_name=self.config.post_function,
            batch_size=self.config.batch_size,
            config_json=self.config.config_json_path,
        )

        # Configure tile cropper with calculated parameters
        # tiling_mode: 0 = single-scale, 1 = multi-scale
        tiling_mode = 1 if self.config.use_multi_scale else 0

        # Set scale_level based on mode
        # Single-scale: scale_level not used (pass 0 to skip in pipeline string)
        # Multi-scale: scale_level 1={1x1}, 2={1x1,2x2}, 3={1x1,2x2,3x3}
        scale_level = self.config.scale_level if self.config.use_multi_scale else 0

        tile_cropper_pipeline = TILE_CROPPER_PIPELINE(
            detection_pipeline,
            name='tile_cropper_wrapper',
            internal_offset=True,
            scale_level=scale_level,
            tiling_mode=tiling_mode,
            tiles_along_x_axis=self.config.tiles_x,
            tiles_along_y_axis=self.config.tiles_y,
            overlap_x_axis=self.config.overlap_x,
            overlap_y_axis=self.config.overlap_y,
            iou_threshold=self.config.iou_threshold,
            border_threshold=self.config.border_threshold
        )

        user_callback_pipeline = USER_CALLBACK_PIPELINE()

        display_pipeline = DISPLAY_PIPELINE(
            video_sink=self.video_sink,
            sync=self.sync,
            show_fps=self.show_fps
        )

        pipeline_string = (
            f'{source_pipeline} ! '
            f'{tile_cropper_pipeline} ! '
            f'{user_callback_pipeline} ! '
            f'{display_pipeline}'
        )

        hailo_logger.debug(f"Pipeline string: {pipeline_string}")
        return pipeline_string


def main() -> None:
    # Create an instance of the user app callback class
    user_data = app_callback_class()
    app_callback = dummy_callback
    app = GStreamerTilingApp(app_callback, user_data)
    app.run()

if __name__ == "__main__":
    print("Starting Hailo Tiling App...")
    main()
