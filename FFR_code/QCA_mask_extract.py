import cv2
import numpy as np

QCA_img=cv2.imread('2nd data\\29563\\AP\\AP_QCA.bmp')
img=QCA_img[:581,84:810,:]

img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

# yellow_lower = np.array([22, 93, 0])
# yellow_upper = np.array([45, 255, 255])

# red_lower = np.array([0, 255, 255])
# red_upper = np.array([27, 255, 255])

# yellow_mask=cv2.inRange(img_hsv,yellow_lower,yellow_upper)
# red_mask=cv2.inRange(img_hsv,red_lower,red_upper)
# res=cv2.bitwise_and(img,mask)

# mask=cv2.bitwise_or(yellow_mask,red_mask)

mask = cv2.inRange(img_hsv,(0,255,255),(180,255,255))

#kernel = np.ones((3, 3), np.uint8)
#res_mask = cv2.dilate(mask, kernel, iterations = 2)

cv2.imshow('img',QCA_img)
cv2.imshow('mask',mask)
#cv2.imshow('res_mask',res_mask)
cv2.waitKey()