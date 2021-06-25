import cv2
import numpy as np
import os
from skimage import morphology 

def nothing(x):
    pass

def QCA_cropping(QCA_img):
    crop_range=[]
    for j in range(QCA_img.shape[1]-1):
        if bool(QCA_img[:581,j,:].sum())^bool(QCA_img[:581,j+1,:].sum()):
            crop_range.append(j+1)

    if len(crop_range)!=2:
        print("BUG :",crop_range)
        crop_range=[84,810]

    return QCA_img[:581,crop_range[0]:crop_range[1],:]

def make_mask(QCA_img,img_512):
    QCA_crop=QCA_cropping(QCA_img)
    img_hsv=cv2.cvtColor(QCA_crop,cv2.COLOR_BGR2HSV) 
    QCA_mask = cv2.inRange(img_hsv,(0,255,255),(180,255,255))

    return QCA_crop,QCA_mask 

def move_mask(QCA_img,img_512,fill_mask=None):
    QCA_resize = cv2.resize(QCA_img,dsize=(0, 0), fx=512/726, fy=512/726)

    k=0
    m=512*512*255
    for i in range(103):
        tmp=np.sum(np.abs(cv2.subtract(img_512[i:i+410,:,:],QCA_resize)))
        if (tmp<m):
            k=i
            m=tmp

    if k==0:
        return np.concatenate((cv2.resize(fill_mask,dsize=(0, 0), fx=512/726, fy=512/726),np.zeros((102,512),dtype=np.uint8)),axis=0)
    elif k==102:
        return np.concatenate((np.zeros((102,512),dtype=np.uint8),cv2.resize(fill_mask,dsize=(0, 0), fx=512/726, fy=512/726)),axis=0)
    else:
        return np.concatenate((np.zeros((k,512),dtype=np.uint8),
        cv2.resize(fill_mask,dsize=(0, 0), fx=512/726, fy=512/726),
        np.zeros((102-k,512),dtype=np.uint8))
        ,axis=0)
        
def contour_in_mask(res_img):
    # res_img : (512,512,3)

    res=np.zeros(res_img.shape,np.uint8)
    gray=cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)
    _,thresh=cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    #L=[]

    for i in range(len(contours)):
        # if(cv2.arcLength(contours[i], True))>200:
        #     L.append(contours[i])
            #print(cv2.contourArea(contours[i]),cv2.arcLength(contours[i], False))
            #cv2.drawContours(res, [contours[i]], 0, (0, 0, 255), -1)
            #cv2.putText(res, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
            #cv2.circle(res,tuple(contours[i][len(contours[i])//2][0]),3,(0,0,255),-1)
        
        cv2.drawContours(res, [contours[i]], 0, (255, 255, 255), 1)

    #print(len(contours))
    return res
num=0
#문제 : 1538839 1892844
patend_ids=os.listdir('../2nd data')
#patend_ids=['1892844']

imges_path = [(os.path.join('..','2nd data',i,'AP','AP_QCA.bmp'),os.path.join('..','2nd data',i,'AP','AP_contrast.bmp')) for i in patend_ids]
# imges_path = [(os.path.join('2nd data',i,'AP','AP_QCA.bmp'),os.path.join('2nd data',i,'AP','AP_reg.png')) for i in patend_ids]
# print(imges_path.pop(7))
# print(imges_path.pop(8))
# print(imges_path.pop(10))
# print(imges_path.pop(11))
# print(imges_path.pop(11))
# print(imges_path.pop(11))


QCA_img=cv2.imread(imges_path[num][0])
img_512=cv2.imread(imges_path[num][1])
print(num,patend_ids[num])

#QCA_img=cv2.imread('2nd data\\29563\\AP\\AP_QCA.bmp')
#img_512=cv2.imread('2nd data\\29563\\AP\\AP_contrast.bmp')

QCA_crop,QCA_mask = make_mask(QCA_img,img_512)
cv2.imshow('QCA_crop_img',QCA_crop)
cv2.imshow('img_512',img_512)

colormap = cv2.COLORMAP_JET
cv2.imshow('pseudo coloring',cv2.applyColorMap(cv2.cvtColor(img_512,cv2.COLOR_BGR2GRAY), colormap))
# lut=np.array(([ 2*i for i in range(128)]+[0]*128)).clip(0,255).astype('uint8')
# cv2.imshow('pseudo coloring',cv2.applyColorMap(cv2.LUT(cv2.cvtColor(img_512,cv2.COLOR_BGR2GRAY),lut), colormap))


cv2.namedWindow('track_bar')
cv2.createTrackbar('kernel size','track_bar',0,100,nothing)
cv2.createTrackbar('iteration','track_bar',0,10,nothing)

remove_window_name =('fill_mask','res_img','check_res','QCA_crop_img','img_512','pseudo coloring')

while(1):
    key = cv2.waitKeyEx(1)
    #print(key)
    kernel_size = cv2.getTrackbarPos('kernel size','track_bar')
    iteration = cv2.getTrackbarPos('iteration','track_bar')
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    fill_mask = cv2.morphologyEx(QCA_mask, cv2.MORPH_CLOSE,kernel, iterations = iteration)
    #cv2.imshow('fill_mask',fill_mask)

    #res_QCA = cv2.bitwise_and(QCA_crop,QCA_crop,mask=fill_mask)
    #cv2.imshow('res_QCA',res_QCA)

    res_mask=move_mask(QCA_crop,img_512,fill_mask)
    #res_mask=np.concatenate((cv2.resize(fill_mask,dsize=(0, 0), fx=512/726, fy=512/726),np.zeros((102,512),dtype=np.uint8)),axis=0)
        #res_mask=np.concatenate((np.zeros((102,512),dtype=np.uint8),cv2.resize(fill_mask,dsize=(0, 0), fx=512/726, fy=512/726)),axis=0)

    cv2.imshow('res_mask',res_mask)

    # result
    res_img = cv2.bitwise_and(img_512,img_512,mask=res_mask) 
    #cv2.imshow("contour_in_mask",contour_in_mask(res_img))
    #out = morphology.medial_axis(res_mask)
    
    #res_img = cv2.bitwise_and(cv2.applyColorMap(cv2.cvtColor(img_512,cv2.COLOR_BGR2GRAY), colormap)
    # ,cv2.applyColorMap(cv2.cvtColor(img_512,cv2.COLOR_BGR2GRAY), colormap),
    # mask=res_mask) 
    

    cv2.imshow('res_img',res_img)

    #cv2.imshow('res_img',cv2.applyColorMap(cv2.cvtColor(res_img,cv2.COLOR_BGR2GRAY), colormap))

    # check
    check_img = np.copy(img_512)
    check_img[(res_mask==255)] = [0,0,255]
    #cv2.imshow('check_res',check_img)

    if  key == 27:
        break
    elif key==122:
        
        for i in remove_window_name:
            cv2.destroyWindow(i)
        num = num-1 if num-1>0 else 0
        print(num,patend_ids[num])
        QCA_img=cv2.imread(imges_path[num][0])
        img_512=cv2.imread(imges_path[num][1])
        QCA_crop,QCA_mask = make_mask(QCA_img,img_512)
        cv2.imshow('QCA_crop_img',QCA_crop)
        cv2.imshow('img_512',img_512)
        cv2.imshow('pseudo coloring',cv2.applyColorMap(cv2.cvtColor(img_512,cv2.COLOR_BGR2GRAY),colormap))

    elif key==120:
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
        cv2.imshow('pseudo coloring',cv2.applyColorMap(cv2.cvtColor(img_512,cv2.COLOR_BGR2GRAY),colormap))

    elif key==32:
        if(os.path.isdir(os.path.join('..',"generated data","QCA_extract_mask"))):
            cv2.imwrite(os.path.join('..',"generated data","QCA_extract_mask",patend_ids[num]+"_QCA_mask"+".png"),res_mask)
            print(num,patend_ids[num],"is saved!")

#cv2.waitKey()

#cv2.imwrite('check_res.png',check_img)
#cv2.imwrite('QCA_img.png',QCA_crop)

