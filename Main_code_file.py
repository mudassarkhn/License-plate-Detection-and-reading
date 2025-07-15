import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import cvzone
import pandas as pd
import ast

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

# Initialize YOLO model
license_plate_model = YOLO("Models/license_plate_model.pt")
cars_model = YOLO('Models/yolo11n.pt')

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Initialize video capture
cap = cv2.VideoCapture('Testing Resources/car2.mp4')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
count=0
# Create VideoWriter object
# output = cv2.VideoWriter('licence_plate_reading 2.mp4', 
#                         cv2.VideoWriter_fourcc(*'mp4v'),
#                         fps,  # Dividing by 3 since we're processing every 3rd frame
#                         (1080, 600))  # Updated to match the resized frame size
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    
    frame = cv2.resize(frame, (1080, 600))
    
    # run 
    # ... existing code ...
# ... existing code ...

    # run 
    license_plate_results = license_plate_model.track(frame, persist=True)
    cars_results = cars_model.track(frame, persist=True, classes=2 ,conf=0.50)
 # Extract license plate details
    license_plate_box = license_plate_results[0].boxes.xyxy.cpu().numpy()
    for lp_box in license_plate_box:
        lp_x1, lp_y1, lp_x2, lp_y2 = map(int, lp_box)
        cropped_plate = frame[lp_y1:lp_y2, lp_x1:lp_x2]
        
        # Apply OCR to the cropped license plate
        ocr_results = ocr.ocr(cropped_plate, cls=True)
        if ocr_results is not None and len(ocr_results) > 0:  # Check if ocr_results is not None and has data
            for result in ocr_results:
                if result is not None:  # Ensure result is not None before iterating
                    for line in result:
                        text = line[1][0]  # Extract the text
                        text = text.replace(" ", "").replace("\n", "").replace('-', '')  # Remove spaces and newlines
                        # Display the text above the bounding box of the car
                        
                    # # Resize cropped plate for display
                    cropped_plate_resized = cv2.resize(cropped_plate, (80, 40))  # Resize as needed
                    # frame[lp_y1 - 60:lp_y1, lp_x1:lp_x1 + cropped_plate_resized.shape[1]] = (255, 255, 255)
                            
                    # # Ensure the placement is within bounds
                    if lp_y1 - 150 >= 0:  # Check if there's enough space above the bounding box
                        # Create a white background above the bounding box
                        frame[lp_y1 - 180:lp_y1 -140, lp_x1:lp_x1 + cropped_plate_resized.shape[1]] = (255, 255, 255)
                        # Place the cropped plate slightly higher on the white background
                        frame[lp_y1 - 140:lp_y1 - 100, lp_x1:lp_x1 + cropped_plate_resized.shape[1]] = cropped_plate_resized  # Adjusted Y-coordinates
                        # Draw text above the white background
                        cv2.putText(frame, text, (lp_x1 + 12, lp_y1 - 155),  # Adjusted Y-coordinate
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
                    else:
                    #     # If not enough space, place it at the top of the bounding box
                        frame[lp_y1:lp_y1 + cropped_plate_resized.shape[0], lp_x1:lp_x1 + cropped_plate_resized.shape[1]] = cropped_plate_resized
                        # Draw text above the white background
                        cv2.putText(frame, text, (lp_x1 + 5, lp_y1 + cropped_plate_resized.shape[0] + 10),  # Adjusted Y-coordinate
                                   0.8, (0, 0, 0), 1)
           
                    # cv2.rectangle(frame, (lp_x1, lp_y1), (lp_x2, lp_y2), (0, 255, 0), 2)  # Green box for license plate
                    draw_border(frame, (int(lp_x1), int(lp_y1)), (int(lp_x2), int(lp_y2)), (0, 255, 0), 2,
                        line_length_x=15, line_length_y=15)

    if cars_results[0].boxes.id is not None:
        boxes = cars_results[0].boxes.xyxy.cpu().numpy()
        track_ids = cars_results[0].boxes.id.cpu().numpy()
        classes = cars_results[0].boxes.cls.cpu().numpy()
        names = cars_results[0].names
        
        for box, track_id, cls in zip(boxes, track_ids, classes):
            x1, y1, x2, y2 = map(int, box)

            # x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2,
                        line_length_x=60, line_length_y=60)
            # Draw bounding box for the car
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box for car

                           
    cv2.imshow("Number Plate Detection", frame)
    # output.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
# output.release()
cv2.destroyAllWindows()
