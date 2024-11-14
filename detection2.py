import cv2
import os

frame_directory = "./"  # Update with the correct path
frame_files = sorted(os.listdir(frame_directory))

for frame_file in frame_files:
    if frame_file.endswith(".png"):  # Ensure you're only processing the frames
        frame = cv2.imread(os.path.join(frame_directory, frame_file))
        
        # Your object detection code goes here
        detections = object_detection_model.detect(frame)

        # Drawing bounding boxes and labels on the frame
        for detection in detections:
            x1, y1, x2, y2, label, confidence = detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame with detections
        cv2.imshow("Object Detection", frame)
        cv2.waitKey(1)
        
        # Optionally save the frame
        cv2.imwrite(f"detected_{frame_file}", frame)

cv2.destroyAllWindows()
