# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 14:53:10 2018

@author: S795641
"""

import cv2
import os
import math

execution_path = os.getcwd()
count = 0
idx = 0
for i in range(3):
    capture = cv2.VideoCapture(i)
    if capture.isOpened():
        idx = i
        break
print("the index of webcam is "+str(idx))
cap = cv2.VideoCapture(idx)
frameRate = cap.set(cv2.CAP_PROP_FPS, 10)
print(frameRate)
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))

while(True):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename ="./frames/rock/frame%d.jpg" % count;count+=1
        cv2.imwrite(filename, frame)
        cv2.imshow('Images', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #if cv2.waitKey(1) & 0xFF == ord('d'):
        #detectObject(pathImg)

cap.release()
#out.release()
cv2.destroyAllWindows()