# 360YOLO
___

A small desktop GUI application that displays a 360° equirectangular video feed and runs real-time YOLO object detection.

The application can work with:

- a **live 360 camera feed**
- a **360 equirectangular MP4 video file** passed at launch time
- a **ROS 2 image topic** generated from a live 360 stream

It also supports recording the **annotated panorama preview** to a **ROS 2 bag** while the application is running.

It runs using Python 3.10 and higher and depends on:

- [imgui-bundle](https://pthom.github.io/imgui_bundle/)
- [opencv-python](https://github.com/opencv/opencv-python/)
- [PyOpenGL](https://mcfletch.github.io/pyopengl/)
- [numpy](https://numpy.org/)
- [Ultralytics](https://github.com/ultralytics/ultralytics)

Author: [**Lucio R. Salinas**](https://www.linkedin.com/in/lucio-r-salinas/)

## Features

- Desktop GUI for viewing a 360° equirectangular panorama
- Real-time YOLO object detection
- Multiple projected views generated from the panorama
- Support for **live camera input**
- Support for **MP4 video file input**
- Support for **ROS 2 image input**
- Support for recording the **annotated panorama preview** to a **ROS 2 bag**
- Simple command-line launch for switching input source

## First-time setup

1. Install **Python** >= **3.10**

   Follow the steps in the [BeginnersGuide/Download](http://wiki.python.org/moin/BeginnersGuide/Download) wiki page, or directly download it from the [official download page](https://www.python.org/downloads/)

2. Install **FFmpeg**

   Download and install **FFmpeg** from the [official download page](https://www.ffmpeg.org/download.html).

   Make sure the FFmpeg executables are available from your `PATH`.

3. Clone the repository using [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

   ```bash
   git clone https://github.com/LucioRS/360YOLO.git
   ```

   Or download the repository as a ZIP archive and extract it.

4. In a terminal, within the `360YOLO` folder, create a virtual environment named `.venv`

   ```bash
   py -m venv .venv
   ```

   On Unix/macOS:

   ```bash
   python3 -m venv .venv
   ```

5. Activate the virtual environment

   Windows (Command Prompt):

   ```bash
   .venv\Scripts\activate
   ```

   Windows (PowerShell):

   ```bash
   .venv\Scripts\Activate.ps1
   ```

   Windows (Git Bash):

   ```bash
   source .venv/Scripts/activate
   ```

   Unix/macOS:

   ```bash
   source .venv/bin/activate
   ```

   Your terminal prompt should display `(.venv)` when the virtual environment is active.

6. Install dependencies

   Standard desktop mode:

   ```bash
   pip install -r requirements.txt
   ```

   ROS 2 mode, or if you want the ROS bag recording capability, use an environment with access to ROS 2 and install:

   ```bash
   pip install -r requirements_ros.txt
   ```

   The `requirements_ros.txt` file is intended for environments where ROS 2 Python packages are already available, such as a ROS-sourced shell or an environment layered on top of a ROS installation.

## Running the app

Run the application from the repository root:

```bash
python app/main.py
```

## Input modes

### 1) Live camera input

If launched without arguments, the application uses the camera source configured in the app settings (`CameraConfig`):

```bash
python app/main.py
```

This mode is intended for live 360 camera capture, such as a RICOH THETA camera.

### 2) Video file input

You can also launch the application using a 360 equirectangular MP4 file as input by passing the file path as a positional argument:

```bash
python app/main.py "./videos/my_360_video.mp4" --width 1920 --height 960
```

When a video file path is provided, the application switches from live camera input to file-based input.

### 3) ROS 2 input

The application can also consume a ROS 2 live stream after it has been republished as a raw `sensor_msgs/msg/Image` topic.

This mode is intended for workflows where a camera node publishes compressed `ffmpeg_image_transport_msgs/msg/FFMPEGPacket` messages and the stream is decoded through `image_transport`.

#### ROS 2 prerequisites

For ROS 2 mode, make sure:

- ROS 2 is installed and sourced in the terminal used to launch the app
- `image_transport` is available
- `ffmpeg_image_transport` is installed
- a republisher node is running to convert the `ffmpeg` transport stream into a raw image topic

#### Republish the ffmpeg transport to a raw image topic

If your camera node publishes to:

```text
/camera/image_h264/ffmpeg
```

run:

```bash
ros2 run image_transport republish --ros-args -p in_transport:=ffmpeg -p out_transport:=raw --remap in/ffmpeg:=/camera/image_h264/ffmpeg --remap out:=/camera/image_decoded
```

This creates a decoded raw image topic:

```text
/camera/image_decoded
```

#### Launch 360YOLO in ROS 2 mode

Then launch the app with the decoded ROS 2 image topic:

```bash
python app/main.py --ros-topic /camera/image_decoded --width 1920 --height 960 --fps 30
```

You can adjust `--width`, `--height`, and `--fps` to match the incoming panorama stream.

## Recording the panorama to a ROS 2 bag

The application can record the **annotated panorama preview** shown in the Panorama panel to a **ROS 2 bag**.

This recording contains the panorama image with its current annotations and is saved as a ROS image topic inside the bag.

### Recording behavior

- Recording is started and stopped from the GUI
- The recorded topic contains the **annotated panorama preview**
- The recording is stored as a ROS 2 bag using **MCAP**
- The recorded panorama can later be replayed with ROS 2 tools or converted to a video file

### Requirements for bag recording

To use this feature, launch the application from an environment with access to ROS 2 Python packages and install:

```bash
pip install -r requirements_ros.txt
```

### Playing back the bag

A recorded bag can be played back using standard ROS 2 bag tools, for example:

```bash
ros2 bag play <bag_directory>
```

### Optional conversion from bag to MP4

If you want to convert the recorded MCAP bag to an MP4 video, you can optionally install:

```bash
pip install mcap-to-mp4
```

Then convert the bag using the recorded panorama topic, for example:

```bash
mcap-to-mp4 <path_to_mcap_file> -t /panorama/annotated -o panorama.mp4
```

This is useful when you want a standard video file for visualization, sharing, or post-processing outside ROS 2.

## Expected video format

For best results, the input panorama should be:

- an **equirectangular 360° image or video**
- resolution **3840x1920** or **1920x960**
- preferably **30 FPS**

Using a different resolution may require adjusting the panorama width and height passed to the application.

## Notes

- FFmpeg must be correctly installed and accessible from the command line.
- Live camera mode, video file mode, and ROS 2 mode share the same GUI and detection pipeline.
- When using a video file or ROS 2 input, the panorama is processed in the same way as a live 360 feed.
- In ROS 2 mode, the app currently subscribes to a decoded raw image topic, not directly to the `ffmpeg` transport topic.
- The ROS 2 mode has been tested with decoded images in `bgr8` encoding.
- The ROS bag recording feature stores the **annotated panorama preview** displayed by the application.
- The recorded bag is useful both for later ROS 2 playback and for optional conversion to MP4.

## Examples

Launch with live camera input:

```bash
python app/main.py
```

Launch with MP4 video input:

```bash
python app/main.py "./videos/my_360_video.mp4" --width 1920 --height 960
```

Launch with ROS 2 input:

```bash
python app/main.py --ros-topic /camera/image_decoded --width 1920 --height 960 --fps 30
```