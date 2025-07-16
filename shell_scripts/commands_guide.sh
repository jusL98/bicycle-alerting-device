# -----------------------------------------------------------------------------------------
# Run Commands
cd bicycle-alerting-device
source .venv/bin/activate

python3 TFLite_image_detection.py --modeldir=Sample_TFLite_model --image=test_images/cars_1     # single image
    #OR
python3 TFLite_image_detection.py --modeldir=Sample_TFLite_model --imagedir=test_images         # full image dir


python3 TFLite_video_detection.py --modeldir=Sample_TFLite_model --video=cars_sample.mp4        # single video
    #OR
python3 TFLite_video_detection.py --modeldir=Sample_TFLite_model --imagedir=test_videos         # full video dir


python3 TFLite_webcam_detection.py --modeldir=Sample_TFLite_model




# -----------------------------------------------------------------------------------------
# Shortcut Commands
./runodimage
./runodvideo
./runodwebcam




# -----------------------------------------------------------------------------------------
# How To Edit/Create Bash Files
sudo nano hello-world.sh

#!/bin/bash
#[type code here]

sudo chmod +x hello-world.sh

./hello-world.sh