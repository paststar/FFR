import cv2
import numpy as np

def QCA_cropping(QCA_img):
    crop_range=[]
    for j in range(QCA_img.shape[1]-1):
        if bool(QCA_img[:581,j,:].sum())^bool(QCA_img[:581,j+1,:].sum()):
            crop_range.append(j+1)
    if len(crop_range)!=2:
        print("BUG :",crop_range)
        crop_range=[84,810]
        
    return QCA_img[:581,crop_range[0]:crop_range[1],:]