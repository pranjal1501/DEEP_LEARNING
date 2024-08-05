import pandas as pd
import torch
import cv2
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

# Define your dataset class for predictions
class PoseDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data.iloc[idx, 1:17].astype('float32').values  # Assuming columns x_5 to x_16 are your features
        return torch.tensor(x, dtype=torch.float32)

# Define your MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load your trained model
input_size = 16  # Assuming 16 features (x_5 to x_16)
hidden_size = 128
num_classes = 3   # Number of classes (sitting, standing, squatting)

model = MLP(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load('E:/posture_detections/neural network/pose_classification_model_85.pth'))
model.eval()

# Load your testing CSV file for predictions
csv_file = "E:/posture_detections/TESTING/FRONTVIEW_TEST/testing_frontview.csv"
predictions_data = pd.read_csv(csv_file)

# Initialize dataset for predictions
dataset = PoseDataset(predictions_data)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# OpenCV Video Capture
video_file = "E:/posture_detections/TESTING/FRONTVIEW_TEST/frontview_test.mp4"
cap = cv2.VideoCapture(video_file)

# Initialize Video Writer for output video
output_file = 'output_video_with_predictions_usecases.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Prepare for storing predictions and frame count
predictions = []
frame_count = 0

# Initialize timer variables
sitting_start_time = None
idle_threshold = 0.2  # seconds

# Perform predictions and overlay on video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform prediction for the current frame
    inputs = dataset[frame_count].unsqueeze(0)  # Get inputs for the current frame
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    predicted_class = predicted.item()
    predictions.append(predicted_class)
    
    # Overlay predicted class name on the frame
    class_name = {0: 'Sitting', 1: 'Standing', 2: 'Squatting'}[predicted_class]
    
    if class_name == 'Sitting':
        if sitting_start_time is None:
            sitting_start_time = time.time()
        sitting_duration = time.time() - sitting_start_time
        if sitting_duration > idle_threshold:
            overlay_text = f'Predicted: {class_name} (Idle: {sitting_duration:.1f}s)'
        else:
            overlay_text = f'Predicted: {class_name} ({sitting_duration:.1f}s)'
    else:
        sitting_start_time = None
        overlay_text = f'Predicted: {class_name}'
    
    cv2.putText(frame, overlay_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Write the frame with overlay to the output video
    out.write(frame)
    
    # Display the frame with overlay (optional)
    cv2.imshow('Frame with Predictions', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

