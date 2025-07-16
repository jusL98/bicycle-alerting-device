<a id="readme-top"></a>

# Bicycle Alerting Device

This program is a bicycle safety system that uses AI powered computer vision to detect vehicles and pedestrians approaching cyclists. Using a lightweight TensorFlow Lite model and real-time camera input, it provides immediate audio alerts when cars or people are detected, helping cyclists stay aware of their surroundings and potential hazards.

The system aims to save lives by cyclists them stay alert of approaching vehicles, addressing the major issue of high cyclist deaths from vehicular collisions. A prototype was developed with Raspberry Pi, incorporating a camera, speaker, Raspberry Pi board & case and rear bicycle mount.

I developed this project for the York Region Science & Technology Fair (2023), winning 2nd overall and best in the Digital Technology category.

The project's code is largely an extension of Evan Juras' [TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/deploy_guides/Windows_TFLite_Guide.md) guide. Thanks Evan!

https://github.com/user-attachments/assets/95ea480c-b6f1-4eb3-866c-d5e6ddedb8eb

## Description

Bicycle Alerting Device is an innovative bicycle safety system designed to enhance cyclist awareness and prevent accidents. It uses TensorFlow Lite's object detection with Google's sample TensorFlow Lite model to detect approaching vehicles and pedestrians in real time. By processing live camera input, the system provides immediate audio alerts, allowing cyclists to react to potential hazards and stay safe.

The system supports multiple detection modes:
- real-time webcam monitoring for live cycling
- video file analysis (for testing and validation)
- image processing (for testing and validation)

## Built With

- [Python 3.13](https://www.python.org/): Programming language for core functionality
- [TensorFlow Lite](https://www.tensorflow.org/lite): Lightweight machine learning framework for object detection
- [OpenCV (cv2)](https://opencv.org/): Computer vision library for video processing and camera input
- [NumPy](https://numpy.org/): Library for creating the GUI
- [Pygame](https://www.pygame.org/): Audio playback for alert sounds

## Quick Start

### Prerequisites

**Hardware**

- Raspberry Pi 4
- Webcam or USB Camera
- Audio Output Device
- Monitor
- Keyboard & Mouse

**Software**

- Raspberry Pi OS
- Terminal
- Git

### Installation

To install Bicycle Alerting Device, follow these steps:

1. Fully update the Raspberry Pi:

   ```bash
   sudo apt-get update
   sudo apt-get dist-upgrade
   ```

2. Ensure the camera is enabled in Preferences -> Raspberry Pi Configuration -> Interfaces -> Camera.

3. Clone the repository:

   ```bash
   git clone https://github.com/jusL98/bicycle-alerting-device.git
   cd bicycle-alerting-device
   ```

4. Create and activate a virtual environment:

   ```bash
   sudo pip3 install virtualenv
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Install the required dependencies (TensorFlow, OpenCV, etc.) - 400MB worth:

   ```bash
   bash pi_requirements.sh
   ```

### Setup

6. Download Google's sample Tensor Flow Lite model:

   ```bash
   wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
   ```

7. Unzip Google's sample Tensor Flow Lite model:
   ```bash
   unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d Sample_TFLite_model
   ```

### Run

8. Run **IMAGE** detection using test images:

   ```bash
   cd bicycle-alerting-device
   source .venv/bin/activate

   python3 TFLite_image_detection.py --modeldir=Sample_TFLite_model --image=test_images/cars_1     # single image
      #OR
   python3 TFLite_image_detection.py --modeldir=Sample_TFLite_model --imagedir=test_images         # full image dir
   ```

   OR convert into bash and execute `runodimage.sh`

   \*\* Click any key to cycle through images

   \*\* Press Q to quit

9. Run **VIDEO** detection using test images:

   ```bash
   cd bicycle-alerting-device
   source .venv/bin/activate

   python3 TFLite_video_detection.py --modeldir=Sample_TFLite_model --video=cars_sample.mp4        # single video
      #OR
   python3 TFLite_video_detection.py --modeldir=Sample_TFLite_model --imagedir=test_videos         # full video dir
   ```

   OR convert into bash and execute `runodvideo.sh`

   \*\* Videos auto cycle

   \*\* Q to quit

10. Run **WEBCAM** detection using test images:

    ```bash
    cd bicycle-alerting-device
    source .venv/bin/activate

    python3 TFLite_webcam_detection.py --modeldir=Sample_TFLite_model
    ```

    OR convert into bash and execute `runodwebcam.sh`

    \*\* Q to quit

## Usage

1. Activate one of the three detection scripts.
2. Watch green boxes and a confidence score highlight people or cars, recognizing them.
3. A "beep" sound alert with sound while the detected object is in frame.
4. For image and video with the `--imagedir` or `--videodir` parameter, press any key to cycle. 
5. Press `Q` to quit.

## Contributing

1. Fork & branch off main.
2. Make your changes.
3. PRs welcome!

## Project Structure

```
├── bicycle-alerting-device/
│   ├── TFLite_image_detection.py      # contains the code for IMAGE detection
│   ├── TFLite_video_detection.py      # contains the code for VIDEO detection
│   ├── TFLite_webcam_detection.py     # contains the code for WEBCAM detection
│   ├── code_snippets                  # folder of testing code and fragmented code
│   ├── shell_scripts                  # shell scripts to be converted to bash to quickly run programs
│   ├── test_images                    # folder of test images for IMAGE detection
│   ├── test_videos                    # folder of test videos for VIDEO detection
│   ├── beep.mp3                       # audio file for "beep" audio alert
│   ├── pi_requirements.sh             # list of required dependencies for easy installation
│   └── Prototype Video Demo.mp4       # video of live demo of Bicycle Alerting Device
```

## Acknowledgements

This project was created for the York Region Science & Technology Fair (2023) with the goal to aid in reducing cyclist deaths from vehicular collisions.

Big thanks to Evan Juras' [TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/deploy_guides/Windows_TFLite_Guide.md) guide and boilerplate code that made this project possible.

## License

This project is licensed under the [MIT](LICENSE.txt) License. See LICENSE.txt for more information.

<br>

---

Thank you!

<p align="left">
  <a href="mailto:justin.matthew.lee.18@gmail.com">
    <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white"/>
  </a>
  <a href="https://www.linkedin.com/in/justin-matthew-lee/">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/>
  </a>
    <a href="https://github.com/jusl98">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white"/>
  </a>
</p>

<p align="right">(<a href="#readme-top">BACK TO TOP</a>)</p>
