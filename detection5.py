import cv2
import numpy as np
import os
import time
import glob

frame_folder = "./"
frames = sorted([f for f in os.listdir(frame_folder) if f.endswith(".png")])

timings = []

lower_bound = np.array([200, 200, 200])
upper_bound = np.array([255, 255, 255])

for frame_name in frames:
    frame_path = os.path.join(frame_folder, frame_name)
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"Failed to load {frame_name}")
        continue

    start_time = time.time()

    mask = cv2.inRange(frame, lower_bound, upper_bound)
    output = cv2.bitwise_and(frame, frame, mask=mask)

    end_time = time.time()
    process_time = end_time - start_time
    timings.append(process_time)

    print(f"{frame_name} processed in {process_time:.4f} seconds")

    # Save the processed frame as an image
    output_frame_path = os.path.join(frame_folder, f"processed_{frame_name}")
    cv2.imwrite(output_frame_path, output)  # Save output frame to disk

avg_time = sum(timings) / len(timings)
print(f"Average processing time per frame: {avg_time:.4f} seconds")
