#overlays keypoints from csv file on video frame by frame
import cv2
import pandas as pd

# Path to the CSV file containing keypoints coordinates
csv_file = "E:posture_detections/TESTING/FRONTVIEW_TEST/testing_frontview.csv"

# Path to the original video file
video_file = "E:posture_detections/TESTING/FRONTVIEW_TEST/frontview_test.mp4"

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file)

# Open the video file
cap = cv2.VideoCapture(video_file)

# Check if video file opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Read and display each frame from the video
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Get the frame number
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    # Check if there are keypoints for the current frame
    if f"Frame_{frame_number}" in df['Frame'].values:
        # Get the keypoints for the current frame
        keypoints_row = df.loc[df['Frame'] == f"Frame_{frame_number}"]
        
        # Extract x and y coordinates from DataFrame
        x_coords = keypoints_row[['x_5', 'x_6', 'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16']].values.flatten()
        y_coords = keypoints_row[['y_5', 'y_6', 'y_11', 'y_12', 'y_13', 'y_14', 'y_15', 'y_16']].values.flatten()
        
        # Plot keypoints on the frame
        for x, y in zip(x_coords, y_coords):
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green circles for keypoints
        
    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Press 'q' on keyboard to exit the video
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
