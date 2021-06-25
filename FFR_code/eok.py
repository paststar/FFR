import os
import cv2
import numpy as np

#A = np.zeros((3,3),dtype=np.uint8)

A = np.array([
    [0,0,0],
    [255,255,255],
    [0,0,0],
],dtype=np.uint8)

cv2.imshow("eok",A)
cv2.waitKey(0)

