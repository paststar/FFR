import cv2
import numpy as np
import os

def nothing(x):
    pass

def show(num):
    #res_mask=cv2.imread(res_mask_path[num])
    #cv2.imshow('res_mask',res_mask)

    QCA_img=cv2.imread(imges_path[num][0])
    img_512=cv2.imread(imges_path[num][1])
    cv2.imshow('img_512',img_512)
    
def make_mask(QCA_img,img_512):
    QCA_crop=QCA_cropping(QCA_img)
    img_hsv=cv2.cvtColor(QCA_crop,cv2.COLOR_BGR2HSV) 
    QCA_mask = cv2.inRange(img_hsv,(0,255,255),(180,255,255))

    return QCA_crop,QCA_mask 

def QCA_cropping(QCA_img):
    crop_range=[]
    for j in range(QCA_img.shape[1]-1):
        if bool(QCA_img[:581,j,:].sum())^bool(QCA_img[:581,j+1,:].sum()):
            crop_range.append(j+1)

    if len(crop_range)!=2:
        print("BUG :",crop_range)
        crop_range=[84,810]

    return QCA_img[:581,crop_range[0]:crop_range[1],:]

num=0
patend_ids=os.listdir('2nd data')
imges_path = [(os.path.join('2nd data',i,'AP','AP_QCA.bmp'),os.path.join('2nd data',i,'AP','AP_contrast.bmp')) for i in patend_ids]

QCA_img=cv2.imread(imges_path[num][0])
img_512=cv2.imread(imges_path[num][1])
print(num,patend_ids[num])

QCA_crop,QCA_mask = make_mask(QCA_img,img_512)
cv2.imshow('QCA_crop_img',QCA_crop)
cv2.imshow('img_512',img_512)

cv2.namedWindow('track_bar')
cv2.createTrackbar('kernel size','track_bar',1,100,nothing)
cv2.createTrackbar('ratio','track_bar',1,100,nothing)

remove_window_name =('fill_mask','res_img','check_res','QCA_crop_img','img_512','pseudo coloring')

while(1):
    key = cv2.waitKeyEx(1)

    kernel_size = cv2.getTrackbarPos('kernel size','track_bar')
    ratio = cv2.getTrackbarPos('ratio','track_bar')/1000

    lsc = cv2.ximgproc.createSuperpixelLSC(img_512,kernel_size,ratio)
    #lsc = cv2.ximgproc.createSuperpixelSLIC(img_512,101,region_size=kernel_size,ruler=ratio*1000)

    lsc.iterate(10)
    mask_lsc = lsc.getLabelContourMask()
    label_lsc = lsc.getLabels()
    #print(label_lsc))
    number_lsc = lsc.getNumberOfSuperpixels()
    mask_inv_lsc = cv2.bitwise_not(mask_lsc)
    img_lsc = img_512.copy()
    img_lsc[mask_lsc==255]=[0,0,255]
    #img_lsc = cv2.bitwise_and(img_512,img_512,mask = mask_inv_lsc)
    cv2.imshow("super_pixel",img_lsc)

    if  key == 27:
        break
    elif key==0x250000:
        
        for i in remove_window_name:
            cv2.destroyWindow(i)
        num = num-1 if num-1>0 else 0
        print(num,patend_ids[num])
        QCA_img=cv2.imread(imges_path[num][0])
        img_512=cv2.imread(imges_path[num][1])
        QCA_crop,QCA_mask = make_mask(QCA_img,img_512)
        cv2.imshow('QCA_crop_img',QCA_crop)
        cv2.imshow('img_512',img_512)
        

    elif key==0x270000:
        for i in remove_window_name:
            cv2.destroyWindow(i)
        num+=1
        assert 0<=num<len(patend_ids)
        print(num,patend_ids[num])
        QCA_img=cv2.imread(imges_path[num][0])
        img_512=cv2.imread(imges_path[num][1])
        QCA_crop,QCA_mask = make_mask(QCA_img,img_512)
        cv2.imshow('QCA_crop_img',QCA_crop)
        cv2.imshow('img_512',img_512)
        

    elif key==32:
        pass
        # if(os.path.isdir(os.path.join("generated data","QCA_extract_mask"))):
        #     cv2.imwrite(os.path.join("generated data","QCA_extract_mask",patend_ids[num]+"_QCA_mask"+".png"),res_mask)
        #     print(num,patend_ids[num],"is saved!")

