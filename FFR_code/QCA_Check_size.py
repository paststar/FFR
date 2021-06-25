import cv2
import numpy as np
#import matplotlib.pyplot as plt

QCA_img=cv2.imread('2nd data\\29563\\AP\\AP_QCA.bmp')
#QCA_img=cv2.imread('2nd data\\190371\\AP\\AP_QCA.bmp')
img =cv2.imread('2nd data\\29563\\AP\\AP_contrast.bmp')

QCA_crop=QCA_img[:581,84:810,:]
test=cv2.resize(QCA_crop,dsize=(0, 0), fx=512/726, fy=512/726)
print(test.shape)

#print(QCA_img.shape)
#for i in range(0,700):
    #print(i,(QCA_img[i,:,:]==np.zeros((QCA_img[i,:,:].shape))).sum())


cv2.imshow('QCA_crop',QCA_crop)
cv2.imshow('test',test)
cv2.imshow('img',img[:410,:,:])
#cv2.cvtColor(img[:410,:,:],cv2.COLOR_BGR2GRAY)==cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)

#print((cv2.cvtColor(img[:410,:,:],cv2.COLOR_BGR2GRAY)==cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)).sum()/(410*512))
#print((cv2.cvtColor(img[:410,:,:],cv2.COLOR_BGR2GRAY)==cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)).shape)
#print(QCA_img[:581,84:810,:].shape)
cv2.waitKey()

# crop shape : (581,726,3)
# crop slicing : [:581,84:810,:]