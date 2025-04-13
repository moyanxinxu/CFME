import os
import sys

import cv2
import numpy as np

from get_image import get_image
from mecfTracker import mecfTracker

sys.path.append("./")
# set your video path
path = ""

cap = get_image(path)

# set init bounding box
bbox = [0, 0, 0, 0]

tracker = mecfTracker()
index = 0

# Create output directory if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0  # To keep track of frame numbers

for frame in cap:
    if index == 0:
        tracker.init(frame, bbox)
        index += 1
    else:
        _, bbox = tracker.update(frame)
        bbox = list(map(int, map(np.round, bbox)))
        cv2.rectangle(
            frame,
            (bbox[0], bbox[1]),
            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
            (255, 255, 255),
            1,
        )
        # Save the frame to the output folder
        output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(output_path, frame)
        frame_count += 1