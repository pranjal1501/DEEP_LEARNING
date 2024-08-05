import sys
import os
sys.path.insert(0, os.path.abspath('E:/code/yolov7'))    #path to yolov7 repositary from github ("https://github.com/WongKinYiu/yolov7")
import csv
import cv2
import numpy as np
import pandas as pd
import torch
import easyocr
import re
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from models.experimental import attempt_load
from datetime import datetime, timedelta

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Load the YOLOv7 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vehicle_model = attempt_load('E:/code/yolov7/yolov7.pt', map_location=device)   #path to yolov7 .pt model
vehicle_model.to(device).eval()

# Load the number plate detection model (replace with your actual number plate detection model loading code)
number_plate_model = attempt_load('E:/code/yolov7/best.pt', map_location=device)   #path to number plate detection model .pt file
number_plate_model.to(device).eval()

# Variables for vehicle tracking
next_vehicle_id = 1  # ID counter for assigning unique IDs to vehicles
tracked_vehicles = {}  # Dictionary to store tracked vehicles and their IDs

# CSV file setup
output_csv_path = 'E:/code/output_data.csv' #path to intermidiate csv file(will be processed further)
csv_file = open(output_csv_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame Number', 'Timestamp', 'Vehicle ID', 'Vehicle BBox', 'Plate BBox', 'Plate Text', 'Plate Confidence', 'Vehicle Image Path', 'Plate Image Path'])

# Function to calculate centroid of a bounding box
def calculate_centroid(x1, y1, x2, y2):
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return (cx, cy)

# Function to correct OCR output based on expected format with detailed debug prints


def correct_ocr_output(text):
    # Remove special characters and spaces
    text = re.sub(r'[^A-Za-z0-9]', '', text)
    
    # Convert the entire text to uppercase
    text = text.upper()

    if len(text) != 10:
        return text  # Return original text if length is not 10

    #dictionary(can be modified)
    dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'L': '4', 'G': '6', 'S': '5', 'Q': '0'}
    dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'L', '6': 'G', '5': 'S', '0': 'Q'}

    corrected_text = list(text)

    # List of positions where letters are expected
    letter_positions = [0, 1, 4, 5]
    # List of positions where numbers are expected
    number_positions = [2, 3, 6, 7, 8, 9]

    # Check if the text has the expected length
    if len(corrected_text) < 10:
        return text  # Return original text if length is not 10
    
    try:
        for pos in letter_positions:
            if corrected_text[pos].isnumeric():
                corrected_text[pos] = dict_int_to_char.get(corrected_text[pos], corrected_text[pos])

        for pos in number_positions:
            if not corrected_text[pos].isnumeric():
                corrected_text[pos] = dict_char_to_int.get(corrected_text[pos], corrected_text[pos])
    except KeyError:
        # If a KeyError occurs, return the original text
        return text

    corrected_text = ''.join(corrected_text)
    return corrected_text


# Function to process the frame and perform detection and tracking
def detect_and_track_vehicles(frame, frame_number, timestamp):
    global next_vehicle_id

    # Preprocess the frame for vehicle detection
    vehicle_img = letterbox(frame, 640, stride=32)[0]
    vehicle_img = vehicle_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
    vehicle_img = np.ascontiguousarray(vehicle_img)
    vehicle_img = torch.from_numpy(vehicle_img).to(device).float()
    vehicle_img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if vehicle_img.ndimension() == 3:
        vehicle_img = vehicle_img.unsqueeze(0)

    # Run vehicle detection model on the frame
    with torch.no_grad():
        pred = vehicle_model(vehicle_img)[0]
        # Filter for classes: car (2), motorcycle (3), bus (5), truck (7), train (6)
        pred = non_max_suppression(pred, 0.25, 0.45, classes=[2, 3, 5, 7, 6])

    # Preprocess the frame for number plate detection
    plate_img = letterbox(frame, 640, stride=32)[0]
    plate_img = plate_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
    plate_img = np.ascontiguousarray(plate_img)
    plate_img = torch.from_numpy(plate_img).to(device).float()
    plate_img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if plate_img.ndimension() == 3:
        plate_img = plate_img.unsqueeze(0)

    # Run number plate detection model on the frame
    with torch.no_grad():
        plate_pred = number_plate_model(plate_img)[0]
        plate_pred = non_max_suppression(plate_pred, 0.25, 0.45, classes=None)  # Filter for all classes

    # Dictionary to store current frame's detected vehicle centroids and number plates
    current_frame_vehicles = {}

    # Iterate over detected vehicles
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(vehicle_img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                # Calculate centroid of the bounding box
                centroid = calculate_centroid(x1, y1, x2, y2)

                # Check if this centroid is close to any existing tracked vehicle(threshold=100 pixels, can be modified)
                matched_vehicle_id = None
                for vehicle_id, (prev_centroid, _) in tracked_vehicles.items():
                    if np.sqrt((centroid[0] - prev_centroid[0])**2 + (centroid[1] - prev_centroid[1])**2) < 100:
                        matched_vehicle_id = vehicle_id
                        break

                if matched_vehicle_id is None:
                    # Assign a new ID to this vehicle
                    matched_vehicle_id = next_vehicle_id
                    next_vehicle_id += 1

                # Update tracked vehicles dictionary with the new centroid and bounding box
                tracked_vehicles[matched_vehicle_id] = (centroid, (x1, y1, x2, y2))
                current_frame_vehicles[matched_vehicle_id] = centroid

                # Draw the bounding box and ID on the frame for vehicles(do not comment out this section of code, as bounding boxes the inside the number plate)
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #cv2.putText(frame, f'ID: {matched_vehicle_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Iterate over detected number plates
    for det in plate_pred:
        if len(det):
            det[:, :4] = scale_coords(plate_img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                # Calculate centroid of the bounding box
                plate_centroid = calculate_centroid(x1, y1, x2, y2)

                # Find the nearest vehicle centroid to associate the number plate with
                nearest_vehicle_id = None
                min_distance = float('inf')
                for vehicle_id, (centroid, _) in tracked_vehicles.items():
                    distance = np.sqrt((plate_centroid[0] - centroid[0])**2 + (plate_centroid[1] - centroid[1])**2)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_vehicle_id = vehicle_id

                # Draw the bounding box and label on the frame for number plates
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'Plate', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Convert the number plate region to grayscale and binary for better OCR
                plate_region = frame[y1:y2, x1:x2]
                gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
                _, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                # Calculate skew angle
                moments = cv2.moments(binary_plate)
                if moments['mu20'] - moments['mu02'] != 0:  # Avoid division by zero
                    skew_angle = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])
                    skew_angle_deg = np.degrees(skew_angle)
                else:
                    skew_angle_deg = 0

                # Calculate the rotation matrix
                (h, w) = plate_region.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, skew_angle_deg, 1.0)

                # Rotate the plate image
                rotated_plate_image = cv2.warpAffine(plate_region, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

                # Use EasyOCR to read the text from the rotated number plate with confidence scores
                result = reader.readtext(rotated_plate_image, detail=1)
                plate_texts = [result_item[1] for result_item in result]  # Extract text from each tuple
                plate_text = ' '.join(plate_texts)  # Join extracted texts into a single string

                plate_confidences = [result_item[2] for result_item in result]
                plate_confidence = sum(plate_confidences) / len(plate_confidences) if plate_confidences else 0

                # Correct the OCR output
                corrected_plate_text = correct_ocr_output(plate_text)

                # Display the original and corrected OCR text on the frame
                cv2.putText(frame, f'OCR: {plate_text}', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(frame, f'Corrected: {corrected_plate_text}', (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                # If a vehicle is detected nearby, associate the number plate with that vehicle
                if nearest_vehicle_id is not None:
                    cv2.putText(frame, f'ID: {nearest_vehicle_id}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Capture images of the vehicle and plate
                vehicle_image = frame[tracked_vehicles[nearest_vehicle_id][1][1]:tracked_vehicles[nearest_vehicle_id][1][3],
                                     tracked_vehicles[nearest_vehicle_id][1][0]:tracked_vehicles[nearest_vehicle_id][1][2]]

                # Save images locally with unique filenames
                vehicle_image_path = f'E:/code/1/{frame_number}_{nearest_vehicle_id}_vehicle.jpg' #path to save vehicle images
                plate_image_path = f'E:/code/plate_images_indian_2/{frame_number}_{nearest_vehicle_id}_plate.jpg' #path to save plate images(not saving right now)
                cv2.imwrite(vehicle_image_path, vehicle_image)

                # Save rotated plate image
                rotated_plate_image_path = f'E:/code/2/{frame_number}_{nearest_vehicle_id}_plate_rotated.jpg' #path to save rotated plaate images
                cv2.imwrite(rotated_plate_image_path, rotated_plate_image)

                # Write the confidence score to the CSV file
                csv_row = [frame_number, timestamp.strftime("%H:%M:%S"), nearest_vehicle_id, tracked_vehicles[nearest_vehicle_id][1], (x1, y1, x2, y2),
                           corrected_plate_text, plate_confidence, vehicle_image_path, rotated_plate_image_path]

                # Only write corrected_plate_text if it has exactly 7 characters
                if len(corrected_plate_text) == 10:
                    csv_writer.writerow(csv_row)

    # Remove old vehicles from tracked_vehicles if they are not present in the current frame
    for vehicle_id in list(tracked_vehicles.keys()):
        if vehicle_id not in current_frame_vehicles:
            del tracked_vehicles[vehicle_id]

    return frame

# Open the input video
input_video_path = 'E:/code/test_video/indian_number_plate_video_2.mp4'#input video path
output_video_path = 'E:/code/output_video.mp4'#output video path
cap = cv2.VideoCapture(input_video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_number = 0
timestamp = datetime.strptime("00:00:00", "%H:%M:%S")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    # Calculate timestamp based on current frame number and FPS
    timestamp += timedelta(seconds=1 / fps)

    # Detect and track vehicles, and detect number plates in the frame
    frame = detect_and_track_vehicles(frame, frame_number, timestamp)

    # Write the frame to the output video
    out.write(frame)

# Release everything if job is finished
cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

input_csv_path = 'E:/code/output_data.csv'#intermediate csv file(will be filtered furthuer)
df = pd.read_csv(input_csv_path, encoding='ISO-8859-1')

# Group by Vehicle ID and Plate Text to get the count of occurrences
plate_counts = df.groupby(['Vehicle ID', 'Plate Text']).size().reset_index(name='Count')

# Merge the counts with the original dataframe to get confidence scores
merged_df = pd.merge(df, plate_counts, on=['Vehicle ID', 'Plate Text'])

# For each Vehicle ID, select the Plate Text with the highest occurrence
def get_max_confidence(group):
    # Select the Plate Text with the highest occurrence
    max_occurrence = group.loc[group['Count'].idxmax()]
    # Filter the group to only include rows with the max occurrence Plate Text
    filtered_group = group[group['Plate Text'] == max_occurrence['Plate Text']]
    # From these, select the row with the highest confidence score
    return filtered_group.loc[filtered_group['Plate Confidence'].idxmax()]

# Apply the function to each group of Vehicle ID
result_df = merged_df.groupby('Vehicle ID').apply(get_max_confidence).reset_index(drop=True)

# Drop the bounding box columns
result_df = result_df.drop(columns=['Vehicle BBox', 'Plate BBox'])

# Save the result to a new CSV file
output_csv_path = 'E:/code/highest_occurrence_plates.csv'#final output csv fine
result_df.to_csv(output_csv_path, index=False)

print(f"Processed data saved to {output_csv_path}")
