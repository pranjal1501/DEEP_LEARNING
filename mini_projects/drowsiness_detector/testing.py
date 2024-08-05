import torch
import numpy as np
import cv2
import os



model=torch.hub.load('ultralytics/yolov5','custom',path='yolov5/runs/train/exp4/weights/best.pt',force_reload=True)

# Now you can use `model` to perform inference on images

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from camera.")
        break

    # Perform detection
    results = model(frame)

    cv2.imshow('YOLO',np.squeeze(results.render()))
    # Check for 'q' key press to exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources after the loop ends
cap.release()
cv2.destroyAllWindows()
