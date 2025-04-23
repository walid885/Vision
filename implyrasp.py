import cv2
import mediapipe as mp
import numpy as np
import time
import requests
import os
from picamera2 import Picamera2

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
    max_results=5,
    category_allowlist=["cup", "bowl", "mug"]  # Filter only for cup-like objects
)
detector = vision.ObjectDetector.create_from_options(options)

# Initialize Pi Camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# For FPS calculation
start_time = time.time()
frame_count = 0
skip_frames = 2
frame_index = 0

try:
    while True:
        # Capture frame from Pi Camera
        frame = picam2.capture_array()
        
        frame_index += 1
        # Skip frames to reduce processing load
        if frame_index % skip_frames != 0:
            cv2.imshow('MediaPipe Cup Detection', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            continue
        
        # Convert to RGB if needed (Pi Camera might return in different format)
        if frame.shape[2] == 3:  # Check if RGB already
            frame_rgb = frame  # Pi Camera often returns in RGB format already
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create a MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detect objects
        detection_result = detector.detect(mp_image)
        
        # Convert to BGR for OpenCV display if needed
        frame_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Process results
        for detection in detection_result.detections:
            # Get bounding box
            bbox = detection.bounding_box
            
            # Get coordinates
            x = bbox.origin_x
            y = bbox.origin_y
            w = bbox.width
            h = bbox.height
            
            # Get category
            category = detection.categories[0]
            category_name = category.category_name
            confidence = round(category.score * 100, 1)
            
            # Draw ellipse instead of rectangle for cup-like objects
            center_x = x + w//2
            center_y = y + h//2
            cv2.ellipse(frame_display, (center_x, center_y), (w//2, h//2), 0, 0, 360, (0, 255, 0), 2)
            
            # Draw handle for coffee cup visual
            handle_x = x + w
            handle_y = y + h//3
            handle_width = w//4
            handle_height = h//2
            cv2.ellipse(frame_display, (handle_x, handle_y + handle_height//2), 
                        (handle_width//2, handle_height//2), 0, 270, 90, (0, 255, 0), 2)
            
            # Label it as coffee cup if it's a cup/mug
            label = "Coffee Cup" if category_name in ["cup", "mug"] else category_name
            
            # Display label and confidence
            cv2.putText(frame_display, f"{label}: {confidence}%", 
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            cv2.putText(frame_display, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame_count = 0
            start_time = time.time()
        
        # Display the frame
        cv2.imshow('MediaPipe Cup Detection', frame_display)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    picam2.stop()
    cv2.destroyAllWindows()