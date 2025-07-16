#!/bin/bash

cd bicycle-alerting-device
source .venv/bin/activate
python3 TFLite_image_detection.py --modeldir=Sample_TFLite_model --imagedir=test_images