import os
import sys

import cv2
import numpy as np

from get_image import get_image
from mecfTracker import mecfTracker

sys.path.append("./")
# set your video path
path = "/root/Documents/CFME/img"

cap = get_image(path)

# set init bounding box
bbox = [13, 233, 36, 38]

tracker = mecfTracker()
index = 0

# Define output video parameters
output_video_path = "output/tracked_video.mp4"
os.makedirs("output", exist_ok=True)

# Get video properties (assuming cap is iterable and provides frames with consistent size)
frame_width = None
frame_height = None
fps = 10  # Set frames per second for the output video

# Initialize VideoWriter later when the first frame is available
video_writer = None

for frame in cap:
    if frame_width is None or frame_height is None:
        frame_height, frame_width = frame.shape[:2]
        video_writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),  # Codec for MP4 format
            fps,
            (frame_width, frame_height),
        )

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
        # Write the frame to the video file
        video_writer.write(frame)

# Release the video writer
if video_writer:
    video_writer.release()