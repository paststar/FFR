import cv2
import numpy as np

def nothing(x):
    pass

QCA_img=cv2.imread('2nd data\\29563\\AP\\AP_QCA.bmp')
img_512=cv2.imread('2nd data\\29563\\AP\\AP_contrast.bmp')

QCA_crop=QCA_img[:581,84:810,:]
#QCA_crop=cv2.resize(QCA_crop,dsize=(0, 0), fx=512/726, fy=512/726)
img_hsv=cv2.cvtColor(QCA_crop,cv2.COLOR_BGR2HSV) 
mask = cv2.inRange(img_hsv,(0,255,255),(180,255,255)) 

#print(np.zeros())
#mask = cv2.GaussianBlur(mask, (3,3), 0) + mask

cv2.imshow('QCA_img',QCA_crop)
cv2.imshow('img',img_512)
cv2.imshow('mask',mask)

cv2.createTrackbar('kernel size','mask',0,100,nothing)
cv2.createTrackbar('iteration','mask',0,10,nothing)

while(1):
    
    kernel_size = cv2.getTrackbarPos('kernel size','mask')
    iteration = cv2.getTrackbarPos('iteration','mask')
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    #res_mask = cv2.dilate(mask, kernel, iterations = iteration)
    fill_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,kernel, iterations = iteration)
    cv2.imshow('fill_mask',fill_mask)
    #res_mask=cv2.subtract(fill_mask,mask)
    res_QCA = cv2.bitwise_and(QCA_crop,QCA_crop,mask=fill_mask)
    cv2.imshow('res_QCA',res_QCA)

    #res_mask = cv2.resize(res_mask,(512,512))
    res_mask=np.concatenate((cv2.resize(fill_mask,dsize=(0, 0), fx=512/726, fy=512/726),np.zeros((102,512),dtype=np.uint8)),axis=0)
    # result
    res_img = cv2.bitwise_and(img_512,img_512,mask=res_mask) 
    cv2.imshow('res_img',res_img)
    # check
    check_img = np.copy(img_512)

    #break
    check_img[(res_mask==255)] = [0,0,255]
    cv2.imshow('check_res',check_img)

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cv2.waitKey()

cv2.imwrite('check_res.png',check_img)
cv2.imwrite('QCA_img.png',QCA_crop)
