"""
   Copyright 2025 Justin Matthew Lee

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

# This file has been modified from its original version by Evan Juras.
# Original copyright (c) Evan Juras under the Apache License, Version 2.0.
# https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/tree/master

"""
This program uses a TensorFlow Lite model to perform object detection on a
video. It draws boxes and scores around the objects of interest in each frame
from the video.
"""

import pygame
from datetime import datetime
import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util
import glob

pygame.mixer.init()
pygame.mixer.music.load("beep.mp3")

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in', required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite', default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt', default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.5)
parser.add_argument('--video', help='Name of the video file', default=None)
parser.add_argument('--videodir', help='Directory containing video files (default: test_videos)', default=None)
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection', action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
VIDEO_NAME = args.video
VIDEO_DIR = args.videodir
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu

# If both a video AND a videodir are specified, throw an error
if (VIDEO_NAME and VIDEO_DIR):
    print('Error! Please only use the --video argument or the --videodir argument, not both. Issue "python TFLite_video_detection.py -h" for help.')
    sys.exit()

# If neither a video or a videodir are specified, default to using 'test.mp4' for video name
if (not VIDEO_NAME and not VIDEO_DIR):
    VIDEO_NAME = 'test_videos/test.mp4'

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'   

# Get path to current working directory
CWD_PATH = os.getcwd()

# Define path to videos and grab all video filenames
if VIDEO_DIR:
    PATH_TO_VIDEOS = os.path.join(CWD_PATH, VIDEO_DIR)
    videos = (glob.glob(PATH_TO_VIDEOS + '/*.mp4') + 
              glob.glob(PATH_TO_VIDEOS + '/*.avi') + 
              glob.glob(PATH_TO_VIDEOS + '/*.mov') + 
              glob.glob(PATH_TO_VIDEOS + '/*.mkv') + 
              glob.glob(PATH_TO_VIDEOS + '/*.wmv'))

elif VIDEO_NAME:
    if os.path.dirname(VIDEO_NAME) == "":  # Just filename
        videos = [os.path.join(CWD_PATH, VIDEO_NAME)]
    else:  # Full/relative path
        videos = [os.path.join(CWD_PATH, VIDEO_NAME)]

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Process videos
print(f"Found {len(videos)} video(s) to process")

quit_all = False
for video_path in videos:
    print(f"Processing video: {video_path}")
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        continue

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        continue

    imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    while(video.isOpened()):
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Acquire frame and resize to expected shape [1xHxWx3]
        ret, frame = video.read()
        if not ret:
          print('Reached the end of the video!')
          break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if labels[int(classes[i])] == "person" or labels[int(classes[i])] == "car":  # limits to only person and car
                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    
                    # Audio Beep
                    print(label)
                    pygame.mixer.music.play()
                    
                    # Write to Log
                    file = open("log_OD","a")
                    file.write(datetime.now().strftime('\n%Y-%m-%d %H:%M:%S') + ' - ' + labels[int(classes[i])] + ' detected')
                

        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, now display the frame
        cv2.imshow('Object detector', frame)
        
        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1
        
        # Press any key to continue to next frame, or press 'q' to quit
        if cv2.waitKey(0) == ord('q'):
            # Write to Log (Separator)
            file = open("log_OD","a")
            file.write('\n-------------------------------------')
            file.close()
            # Set flag to quit all videos
            quit_all = True
            break
                
    # Clean up current video
    video.release()
    
    if quit_all:
        break

# Final cleanup
cv2.destroyAllWindows()
