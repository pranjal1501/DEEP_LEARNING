#extract keypoints from input video
from ultralytics import YOLO
import csv

# Load the model
model = YOLO("yolov8x-pose-p6.pt")

# Set the source (use raw string to handle backslashes)
source = "E:/posture_detections/TESTING/FRONTVIEW_TEST/testing_2.mp4"#path to input video

# Define the keypoint indices of interest
keypoints_indices = [5, 6, 11, 12, 13, 14, 15, 16]

# Open a CSV file to write the keypoints
with open('keypoints_map.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    header = ["Frame"] + [f"x_{i}" for i in keypoints_indices] + [f"y_{i}" for i in keypoints_indices]
    writer.writerow(header)
    
    # Perform the prediction with streaming enabled
    frame_number = 0  # Initialize frame number
    
    for result in model.predict(source, save=True, imgsz=320, conf=0.5, stream=True):
        frame_number += 1  # Increment frame number
        
        # Check if there are keypoints
        if hasattr(result, 'keypoints'):
            keypoints_list = result.keypoints.xy.tolist()
            
            for keypoints in keypoints_list:
                # Filter keypoints of interest
                keypoints_of_interest = [keypoints[idx] for idx in keypoints_indices]
                
                # Prepare lists for x and y coordinates
                x_coords = [kp[0] for kp in keypoints_of_interest]
                y_coords = [kp[1] for kp in keypoints_of_interest]
                
                # Create a row for the CSV
                row_data = [f"Frame_{frame_number}"] + x_coords + y_coords
                
                # Write the row to CSV
                writer.writerow(row_data)