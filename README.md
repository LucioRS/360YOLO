# 360YOLO
___
A small desktop GUI application that displays a live 360° camera feed and runs real-time YOLO object detection.

It runs using Python 3.10 and higher and depends on:

- [imgui-bundle](https://pthom.github.io/imgui_bundle/)
- [opencv-python](https://github.com/opencv/opencv-python/)
- [PyOpenGL](https://mcfletch.github.io/pyopengl/)
- [numPy](https://numpy.org/)
- [Ultralytics ](https://github.com/ultralytics/ultralytics)

Author: [**Lucio R. Salinas*](https://www.linkedin.com/in/lucio-r-salinas/)

## First-time setup

1. Install **Python** >= **3.10**

    Follow the steps in the [BeginnersGuide/Download](http://wiki.python.org/moin/BeginnersGuide/Download) wiki page, or directly download it from the [official download page](https://www.python.org/downloads/)

2. Install **FFmpeg**

    Download (https://www.ffmpeg.org/download.html) and install **FFmpeg**. Make sure the executables' folder is on your PATH. 
    
3. Clone the repository using [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git):

   - `git clone https://github.com/LucioRS/360YOLO.git`

    or download the repository as a ZIP archive and extract it

4. In a terminal, within the `360YOLO` folder, create a virtual environment. Call the virtual environment folder `.venv`

	- `py -m venv .venv` (Windows)
	- `python3 -m venv .venv` (Unix/macOS)
	
5. Activate your virtual environment

	- `.venv\Scripts\activate` (Windows, command line)
	- `.venv\Scripts\activate.ps1` (Windows, PowerShell)
    - `source .venv/Scripts/activate` (Windows, Git Bash)
	- `source .venv/bin/activate` (Unix/macOS)

    Your prompt will display `(.venv)` when the virtual environment is active
	
6. Install dependencies

    ```
    pip install -r requirements.txt
    ```

## Running the app

Whithin the active virtual environment, simply run `main.py`.
```
python main.py
```