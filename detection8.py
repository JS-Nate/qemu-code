import cv2
import numpy as np
import os
import time

frame_folder = "./"
frames = sorted([f for f in os.listdir(frame_folder) if f.endswith(".png")])

timings = []

# Define color bounds for object detection
lower_bound = np.array([200, 200, 200])
upper_bound = np.array([255, 255, 255])

# Get dimensions of the first frame for setting up the video writer
first_frame_path = os.path.join(frame_folder, frames[0])
first_frame = cv2.imread(first_frame_path)
frame_height, frame_width, _ = first_frame.shape

# Define video writer (use appropriate codec and file name)
output_video_path = "output_video.avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec
fps = 30  # Frames per second
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

for frame_name in frames:
    frame_path = os.path.join(frame_folder, frame_name)
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"Failed to load {frame_name}")
        continue

    start_time = time.time()

    # Create mask for objects in the specified color range
    mask = cv2.inRange(frame, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours and label detected objects on the frame
    for i, contour in enumerate(contours):
        # Get the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Draw a rectangle around the detected object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Label the object
        label = f"Object {i+1}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Write the processed frame to the video
    video_writer.write(frame)

    end_time = time.time()
    process_time = end_time - start_time
    timings.append(process_time)

    print(f"{frame_name} processed in {process_time:.4f} seconds")
    print(f"Detected {len(contours)} objects in {frame_name}")

# Release the video writer
video_writer.release()

avg_time = sum(timings) / len(timings)
print(f"Average processing time per frame: {avg_time:.4f} seconds")
print(f"Output video saved as {output_video_path}")
