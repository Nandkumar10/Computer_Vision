"""Task: Vehicle tracker and counter:
Create a vehicle tracker using YOLOv8 for object detection and integrate deepsort for tracking.
Keep count of the vehicles going from one side to another side of the road.
Display this count on the inference screen."""

# Import necessary libraries

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  #YOLOv8 model path

# Initialize DeepSort tracker
tracker = DeepSort(max_age=30, nn_budget=100)

# Open video file or webcam
#cap = cv2.VideoCapture("/Users/nandkumaradmane/Documents/GenerativeAI/CV/Test_Video.mp4")  # Update with video path
cap = cv2.VideoCapture("/Users/nandkumaradmane/Documents/GenerativeAI/CV/Video.mp4")  # Update with video path


# Minimum width and height for detected vehicles
min_width_rect = 80
min_height_rect = 80

# Define the counting line position
count_line_position = 550

# Counter for vehicles
vehicle_count = 0
tracked_centers = {}  # To store center points of tracked vehicles

# Vehicle classes in YOLO (COCO dataset: 2=car, 5=bus, 7=truck)
vehicle_classes = [2, 5, 7]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLOv8 detection
    results = model(frame)

    # Prepare detections for DeepSort
    detections = []
    for result in results:
        for box in result.boxes:
            x, y, w, h = box.xywh[0].cpu().numpy()
            conf = box.conf.cpu().numpy()
            class_id = int(box.cls.cpu().numpy())

            # Filter for vehicle classes
            if class_id in vehicle_classes:
                # Convert to DeepSort format: [left, top, width, height], confidence
                detections.append(([x - w/2, y - h/2, w, h], conf, class_id))

    # Update DeepSort tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Process each tracked object
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        bbox = track.to_ltrb()  # Get bounding box as [left, top, right, bottom]
        x, y, w, h = [int(v) for v in [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2,
                                       bbox[2] - bbox[0], bbox[3] - bbox[1]]]

        # Validate size of detection
        if w < min_width_rect or h < min_height_rect:
            continue

        # Calculate center of the bounding box
        center_y = y + h // 2

        # Track the center point for counting
        if track_id not in tracked_centers:
            tracked_centers[track_id] = []

        tracked_centers[track_id].append(center_y)

        # Check if vehicle crosses the counting line
        if len(tracked_centers[track_id]) > 1:
            prev_y = tracked_centers[track_id][-2]
            curr_y = tracked_centers[track_id][-1]

            # Vehicle crosses the line
            if prev_y < count_line_position <= curr_y:
                vehicle_count += 1

        # Draw bounding box and track ID
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Draw the counting line
    cv2.line(frame, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)

    # Display the vehicle count on the frame
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show the frame
    cv2.imshow("Vehicle Tracker", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()