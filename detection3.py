import cv2
import numpy as np
import os
import time
import glob

# Define the frame folder (update path if needed)
frame_folder = "./"  # Adjust the path if your frames are in a different folder

# Get a sorted list of all PNG files in the frame folder
frames = sorted([f for f in os.listdir(frame_folder) if f.endswith(".png")])

# Create an array to store processing times for each frame
timings = []

# Set color range for object detection (white color for example)
lower_bound = np.array([200, 200, 200])  # Adjust values as needed
upper_bound = np.array([255, 255, 255])  # Adjust values as needed

# Loop through each frame
for frame_name in frames:
    frame_path = os.path.join(frame_folder, frame_name)
    frame = cv2.imread(frame_path)  # Read the image file
    
    if frame is None:
        print(f"Failed to load {frame_name}")
        continue
    
    start_time = time.time()  # Start time to measure frame processing
    
    # Create a mask for object detection based on color range
    mask = cv2.inRange(frame, lower_bound, upper_bound)
    
    # Apply the mask to the frame
    output = cv2.bitwise_and(frame, frame, mask=mask)
    
    # End time to calculate processing time
    end_time = time.time()
    
    process_time = end_time - start_time
    timings.append(process_time)
    
    print(f"{frame_name} processed in {process_time:.4f} seconds")
    
    # Show the result with the mask applied (Optional)
    # You can also add the detection result to a window
    cv2.imshow("Detected Frame", output)  # Shows the processed frame
    cv2.waitKey(1)  # Wait for 1 ms to show the image
    
# Calculate the average processing time
avg_time = sum(timings) / len(timings)
print(f"Average processing time per frame: {avg_time:.4f} seconds")

cv2.destroyAllWindows()  # Close any OpenCV windows
