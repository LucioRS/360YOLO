# 360YOLO
___

A small desktop GUI application that displays a 360° equirectangular video feed and runs real-time YOLO object detection.

The application can work with either:

- a **live 360 camera feed**
- a **360 equirectangular MP4 video file** passed at launch time

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

   ```bash
   pip install -r requirements.txt
   ```

## Running the app

Run the application from the repository root:

```bash
python app/main.py
```

## Input modes

### 1) Live camera input

If launched without arguments, the application uses the camera source configured in the app settings:

```bash
python app/main.py
```

This mode is intended for live 360 camera capture, such as a RICOH THETA camera.

### 2) Video file input

You can also launch the application using a 360 equirectangular MP4 file as input by passing the file path as a positional argument:

```bash
python app/main.py "./videos/my_360_video.mp4"
```

When a video file path is provided, the application switches from live camera input to file-based input.

## Expected video format

For best results, the input video file should be:

- an **equirectangular 360° video**
- resolution **3840x1920** or **1920x960**
- preferably **30 FPS**

Using a video with a different resolution may require adjusting the configured panorama width and height in the application.

## Notes

- FFmpeg must be correctly installed and accessible from the command line.
- Live camera mode and video file mode share the same GUI and detection pipeline.
- When using a video file, the panorama is processed in the same way as a live 360 feed.