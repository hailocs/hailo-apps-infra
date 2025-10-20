# Tiling Application

![Depth Example](../../../../doc/images/tiling.png)

#### Run the tiling example:
```bash
hailo-tiling
```
To close the application, press `Ctrl+C`.

Tiling pipeline demonstrates splitting each frame into several tiles which are processed independently by `hailonet` element. This method is especially effective for detecting small objects in high-resolution frames.

There is an option for single-scale or Multi-scale tiling:

Multi-scale demonstrates a case where the video and the training dataset includes objects in different sizes. Dividing the frame to small tiles might miss large objects or “cut" them to small objects. The solution is to split each frame into number of scales (layers) each includes several tiles.

Multi-scale tiling strategy also allows us to filter the correct detection over several scales. For example - use 3 sets of tiles at 3 different scales:

*   Large scale, one tile to cover the entire frame (1x1)
*   Medium scale dividing the frame to 2x2 tiles.
*   Small scale dividing the frame to 3x3 tiles.

In this mode we use 1 + 4 + 9 = 14 tiles for each frame. We can simplify the process by highlighting the main tasks: Crop -> Inference -> Ppost-process -> Aggregate → Remove exceeded boxes → Remove large landscape → Perform NMS

#### Specific arguments:

*   `--tiles-x-axis` optional (default 4) - set number of tiles along x axis (columns)
*   `--tiles-y-axis` optional (default 3) - set number of tiles along y axis (rows)
*   `--overlap-x-axis` optional (default 0.1) - set overlap in percentage between tiles along x axis (columns)
*   `--overlap-y-axis` optional (default 0.08) - set overlap in percentage between tiles along y axis (rows)
*   `--iou-threshold` optional (default 0.3) - set iou threshold for NMS.
*   `--border-threshold` optional (default 0.1) - set border threshold to Remove tile's exceeded objects.
*   `--single_scaling` optional (default not provided -> multi-scale) - If provided - use single scaling or multi scaling.
*   `--scale-level` optional (default 2, only for multi-scale) - set scales (layers of tiles) in addition to the main layer. 1: [(1 X 1)] 2: [(1 X 1), (2 X 2)] 3: [(1 X 1), (2 X 2), (3 X 3)]'

#### Running with Raspberry Pi Camera input:
```bash
hailo-tiling --input rpi
```

#### Running with USB camera input (webcam):
There are 2 ways:

Specify the argument `--input` to `usb`:
```bash
hailo-tiling --input usb
```

This will automatically detect the available USB camera (if multiple are connected, it will use the first detected).

Second way:

Detect the available camera using this script:
```bash
get-usb-camera
```
Run example using USB camera input - Use the device found by the previous script:
```bash
hailo-tiling --input /dev/video<X>
```

For additional options, execute:
```bash
hailo-tiling --help
```

#### Running as Python script

For examples: 
```bash
python tiling.py --input usb
```

### All pipeline commands support these common arguments:

[Common arguments](../../../../doc/user_guide/running_applications.md#command-line-argument-reference)