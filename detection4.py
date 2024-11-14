import cv2
import numpy as np
import os
import time
import glob

# Set the frame folder (adjust path if needed)
frame_folder = "./"
frames = sorted([f for f in os.listdir(frame_folder) if f.endswith(".png")])

timings = []

# Define the color range for object detection (white color in this case)
lower_bound = np.array([200, 200, 200])
upper_bound = np.array([255, 255, 255])

for frame_name in frames:
    frame_path = os.path.join(frame_folder, frame_name)
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"Failed to load {frame_name}")
        continue

    start_time = time.time()

    # Create a mask for the white objects based on the color range
    mask = cv2.inRange(frame, lower_bound, upper_bound)

    # Apply the mask to the frame
    output = cv2.bitwise_and(frame, frame, mask=mask)

    end_time = time.time()
    process_time = end_time - start_time
    timings.append(process_time)

    print(f"{frame_name} processed in {process_time:.4f} seconds")

    # Display the processed frame with the detection mask applied
    cv2.imshow("Object Detection", output)  # Shows the processed frame
    cv2.waitKey(1)  # Wait for 1 ms to update the image window

# Calculate and print the average processing time per frame
avg_time = sum(timings) / len(timings)
print(f"Average processing time per frame: {avg_time:.4f} seconds")

cv2.destroyAllWindows()  # Close any OpenCV windows after the loop is complete
