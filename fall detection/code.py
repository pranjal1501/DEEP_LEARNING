import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load the YOLOv8 models
fall_detection_model = YOLO('E:/falling_with_yolo/yolov8x.pt')
pose_estimation_model = YOLO("E:/falling_with_yolo/yolov8x-pose-p6.pt")#path to yolo-pose

# Define a function to classify the person's state based on the aspect ratio, height change rate, and width change rate(can be modified)
def classify_state(aspect_ratio, height_change_rate, width_change_rate, was_falling):
    if not was_falling:
        if height_change_rate < -5.0:  # Threshold for height decrease
            if width_change_rate > 3.0:
                return "Falling"
            elif width_change_rate >= 0:
                return "Cropped"
        if aspect_ratio > 1.25:  # Threshold for "Standing" without falling
            return "Standing"
        else:
            return "Fallen"
    if was_falling:
        if aspect_ratio > 1.5:  # Threshold for "Standing" after falling
            return "Standing"
        else:
            return "Fallen"

# Define a function to check if points are nearly aligned using average x-coordinates
def are_nearly_aligned(p1_avg, p2_avg, threshold=0.05):#0.05 is default value
    return abs(p1_avg - p2_avg) < threshold

# Define a function to run the pose estimation model and get the overall status
def get_pose_estimation_status(image):
    frame_height, frame_width = image.shape[:2]
    results = pose_estimation_model(image)
    statuses = []
    color = (0, 0, 0)  # Black color for unknown
    font_scale = 1.0   # Increased font scale for bigger text

    # Flag to check if any keypoints were detected
    keypoints_detected = False

    # Check if there are keypoints
    if hasattr(results[0], 'keypoints'):
        keypoints_list = results[0].keypoints.xy.tolist()
        for keypoints in keypoints_list:
            if len(keypoints) >= 15:
                shoulder_keypoints = [keypoints[idx] for idx in [5, 6] if keypoints[idx] is not None]
                waist_keypoints = [keypoints[idx] for idx in [11, 12] if keypoints[idx] is not None]
                knee_keypoints = [keypoints[idx] for idx in [13, 14] if keypoints[idx] is not None]

                if len(shoulder_keypoints) == 2 and len(waist_keypoints) == 2 and len(knee_keypoints) == 2:
                    keypoints_detected = True
                    normalized_shoulders = [kp[1] / frame_height for kp in shoulder_keypoints]
                    normalized_waists = [kp[1] / frame_height for kp in waist_keypoints]
                    normalized_knees = [kp[1] / frame_height for kp in knee_keypoints]
                    normalized_shoulder_x = [kp[0] / frame_width for kp in shoulder_keypoints]
                    normalized_waist_x = [kp[0] / frame_width for kp in waist_keypoints]
                    normalized_knee_x = [kp[0] / frame_width for kp in knee_keypoints]

                    avg_shoulder_y = sum(normalized_shoulders) / len(normalized_shoulders)
                    avg_waist_y = sum(normalized_waists) / len(normalized_waists)
                    avg_knee_y = sum(normalized_knees) / len(normalized_knees)
                    avg_shoulder_x = sum(normalized_shoulder_x) / len(normalized_shoulder_x)
                    avg_waist_x = sum(normalized_waist_x) / len(normalized_waist_x)
                    avg_knee_x = sum(normalized_knee_x) / len(normalized_knee_x)

                    waist_knee_aligned = are_nearly_aligned(avg_waist_x, avg_knee_x, threshold=0.03)
                    shoulder_waist_aligned = are_nearly_aligned(avg_shoulder_x, avg_waist_x, threshold=0.03)
                    shoulder_above_waist = avg_shoulder_y < avg_waist_y

                    if (waist_knee_aligned or shoulder_waist_aligned) and shoulder_above_waist:
                        status = "standing"
                        color = (0, 255, 0)  # Green color for standing
                    else:
                        status = "fallen"
                        color = (0, 0, 255)  # Red color for fallen

                    statuses.append(status)
                for kp in shoulder_keypoints + waist_keypoints + knee_keypoints:
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(image, (x, y), 5, color, -1)
                if shoulder_keypoints:
                    x, y = int(shoulder_keypoints[0][0]), int(shoulder_keypoints[0][1])
                else:
                    x, y = 50, 50
                cv2.putText(image, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
    if "fallen" in statuses:
        return "fallen", image
    elif "standing" in statuses:
        return "standing", image
    else:
        return "unknown", image

# Open the input video
input_video_path = 'E:/falling_with_yolo/testing/testing_4.mp4'
output_video_path = 'E:/football_analysis/output.mp4' #output video file

cap = cv2.VideoCapture(input_video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Initialize dictionaries to store the previous heights, widths, centers, and states of detected persons
previous_heights = {}
previous_widths = {}
previous_centers = {}
falling_status = {}  # To track if a person has ever been detected as falling
cropped_status = {}  # To track if a person has ever been detected as cropped

frame_index = 0  # Initialize frame index

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 model on the frame
    results = fall_detection_model(frame)

    # Temporary dictionaries to store current frame's centers, heights, and widths
    current_centers = {}
    current_heights = {}
    current_widths = {}

    # Iterate over detected objects
    for result in results:
        boxes = result.boxes  # Bounding boxes
        for box in boxes:
            cls = int(box.cls[0])  # Class index
            if cls == 0:  # class 0 is 'person' in COCO dataset
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

                # Calculate the center of the bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center = (center_x, center_y)

                # Calculate the aspect ratio
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                aspect_ratio = bbox_height / bbox_width

                # Find the closest previous center
                min_distance = float('inf')
                closest_center = None
                for prev_center in previous_centers.keys():
                    distance = np.sqrt((center_x - prev_center[0])**2 + (center_y - prev_center[1])**2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_center = prev_center

                # Calculate height and width change rates
                height_change_rate = 0
                width_change_rate = 0
                if closest_center and min_distance < 100:  # Threshold for considering it the same person(in pixels)
                    height_change_rate = bbox_height - previous_heights[closest_center]
                    width_change_rate = bbox_width - previous_widths[closest_center]
                    was_falling = falling_status.get(closest_center, False)
                    was_cropped = cropped_status.get(closest_center, False)
                    falling_status[center] = was_falling
                    cropped_status[center] = was_cropped
                else:
                    was_falling = False
                    was_cropped = False
                    falling_status[center] = was_falling
                    cropped_status[center] = was_cropped

                previous_heights[center] = bbox_height
                previous_widths[center] = bbox_width

                # Update current frame's centers, heights, and widths
                current_centers[center] = center
                current_heights[center] = bbox_height
                current_widths[center] = bbox_width

                # Classify the state
                state = classify_state(aspect_ratio, height_change_rate, width_change_rate, was_falling)
                if state == "Falling":
                    falling_status[center] = True  # Mark this person as currently falling
                if state == "Cropped":
                    cropped_status[center] = True  # Mark this person as having been cropped

                # Determine color based on the state
                color = (0, 0, 0)  # Default to black
                
                if state == "Standing":
                    color = (0, 255, 0)  # Green for standing
                elif falling_status[center] == True:
                    color = (128, 0, 128)  # purple for falling
                elif state == "Fallen":
                    color = (0, 0, 255)  # Red for fallen
                elif cropped_status[center]:  # green if the person has ever been cropped
                    color = (0, 255, 0)  # green

                # Run pose estimation model on the bounding box if person is classified as "Fallen"
                if state == "Fallen" and falling_status[center] != True:
                    fallen_image = frame[y1:y2, x1:x2]
                    overall_status, annotated_image = get_pose_estimation_status(fallen_image)
                    # Update the state with the result of the second model
                    state = overall_status.capitalize()
                    color = (0, 0, 255) if state == "Fallen" else (0, 255, 0)  # Update color based on new state

                    # Draw the updated bounding box and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, state, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                else:
                    # Draw the bounding box and label on the frame if pose estimation model is not used
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, state, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Update previous centers, heights, and widths for the next frame
    previous_centers = current_centers
    previous_heights = current_heights
    previous_widths = current_widths

    # Write the frame to the output video
    out.write(frame)

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
