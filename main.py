import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from paddleocr import PaddleOCR
import os
from datetime import datetime

# Initialize PaddleOCR
ocr = PaddleOCR()

# Function to perform OCR on an image array
def perform_ocr(image_array):
    if image_array is None:
        raise ValueError("Image is None")
    
    # Perform OCR on the image array
    results = ocr.ocr(image_array, rec=True)  # rec=True enables text recognition
    detected_text = []

    # Process OCR results
    if results[0] is not None:
        for result in results[0]:
            text = result[1][0]
            detected_text.append(text)

    # Join all detected texts into a single string
    return ''.join(detected_text)

# Mouse callback function for RGB window
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load YOLOv8 model
model = YOLO("best.pt")
names = model.names

# Define polygon area
area = [(1, 173), (62, 468), (608, 431), (364, 155)]

# Create directory for current date
current_date = datetime.now().strftime('%Y-%m-%d')
if not os.path.exists(current_date):
    os.makedirs(current_date)

# Create a text file for the current date
text_file_path = os.path.join(current_date, f"{current_date}.txt")
if not os.path.exists(text_file_path):
    with open(text_file_path, "w") as f:
        f.write("Number Plate, Date, Time\n")  # Write headers

# Track processed track IDs
processed_track_ids = set()

# Open the video file or webcam
cap = cv2.VideoCapture('final.mp4')

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (1020, 500))
    
    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True)
    
    # Initialize flags and variables
    no_helmet_detected = False
    numberplate_box = None
    numberplate_track_id = None
    
    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the boxes, class IDs, track IDs, and confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence scores
        
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = names[class_id]
            x1, y1, x2, y2 = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
            if result >= 0:
                if c == 'no-helmet':
                    no_helmet_detected = True  # Mark that no-helmet is detected
                elif c == 'numberplate':
                    numberplate_box = box  # Store the numberplate bounding box
                    numberplate_track_id = track_id  # Store the track ID for the numberplate
        
        # If both no-helmet and numberplate are detected and the track ID is not already processed
        if no_helmet_detected and numberplate_box is not None and numberplate_track_id not in processed_track_ids:
            x1, y1, x2, y2 = numberplate_box
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (120, 75))
            
            # Perform OCR on the cropped image
            text = perform_ocr(crop)
            print(f"Detected Number Plate: {text}")
            
            # Save the cropped image with current time as filename
            current_time = datetime.now().strftime('%H-%M-%S')
            filename = os.path.join(current_date, f'{current_time}_numberplate.jpg')
            cv2.imwrite(filename, crop)
            
            # Save number plate, date, and time to the text file
            with open(text_file_path, "a") as f:
                f.write(f"{text}, {current_date}, {current_time}\n")
            
            # Add the track ID to the processed set
            processed_track_ids.add(numberplate_track_id)
    
    # Draw the polygon
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 255), 2)
    
    # Display the frame
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
