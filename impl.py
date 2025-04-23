import cv2
import mediapipe as mp
import numpy as np
import time
import requests
import os

# Download model if it doesn't exist
model_path = "efficientdet_lite0.tflite"
if not os.path.exists(model_path):
    print("Downloading model...")
    url = "https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite0_uint8.tflite"
    response = requests.get(url)
    with open(model_path, "wb") as f:
        f.write(response.content)
    print("Model downloaded!")

# Import MediaPipe Tasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize object detector
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,
    max_results=5
)
detector = vision.ObjectDetector.create_from_options(options)

# Initialize webcam
cap = cv2.VideoCapture(0)

# For FPS calculation
start_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create a MediaPipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    # Detect objects
    detection_result = detector.detect(mp_image)
    
    # Process results
    for detection in detection_result.detections:
        # Get bounding box
        bbox = detection.bounding_box
        
        # Draw rectangle
        cv2.rectangle(frame, 
                     (bbox.origin_x, bbox.origin_y), 
                     (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), 
                     (0, 255, 0), 2)
        
        # Display category and score
        category = detection.categories[0]
        category_name = category.category_name
        confidence = round(category.score * 100, 1)
        
        cv2.putText(frame, f"{category_name}: {confidence}%", 
                    (bbox.origin_x, bbox.origin_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Calculate and display FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame_count = 0
        start_time = time.time()
    
    # Display the frame
    cv2.imshow('MediaPipe Object Detection', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()