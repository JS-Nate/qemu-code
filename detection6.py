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

    # Create mask for objects in the specified color range
    mask = cv2.inRange(frame, lower_bound, upper_bound)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Count and label the detected objects
    object_count = len(contours)

    # Draw contours and label objects in the original frame
    for i, contour in enumerate(contours):
        # Get the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw a rectangle around the detected object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Label the object
        label = f"Object {i+1}"  # Label each object as "Object 1", "Object 2", etc.
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    end_time = time.time()
    process_time = end_time - start_time
    timings.append(process_time)

    print(f"{frame_name} processed in {process_time:.4f} seconds")
    print(f"Detected {object_count} objects in {frame_name}")

    # Optional: Save the output frame with drawn contours and labels
    output_frame_path = os.path.join(frame_folder, f"processed_{frame_name}")
    cv2.imwrite(output_frame_path, frame)  # Save output frame to disk

avg_time = sum(timings) / len(timings)
print(f"Average processing time per frame: {avg_time:.4f} seconds")
