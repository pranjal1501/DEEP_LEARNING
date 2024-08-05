import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2


import uuid #unique identifier
import os 
import time

from ultralytics import YOLO
model=torch.hub.load('ultralytics/yolov5','yolov5s')

#img_path="C:/Users/pranj/Downloads/image_1.jpg"
#result=model.predict(img_path,save=True)


image_path=os.path.join('data','images') #data->images
labels=['awake','dwrosy']
number_images=30

cap=cv2.VideoCapture(0)


for label in labels:

    print('collecting images for {}'.format(label))
    time.sleep(5)
    for img_num in range(number_images):
        print('collecting images for {} ,images number {}'.format(label,img_num))

        ret,frame=cap.read()

        imgname=os.path.join ( image_path, label+'.'+str(uuid.uuid1()) +'.jpg') #unique imagename
        cv2.imwrite(imgname,frame) #write image to file

        
        if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            print("Error: Failed to load/capture frame or frame dimensions are zero.")
        else:
            cv2.imshow('IMAGE COLLECTION', frame)


        time.sleep(2)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()




                          