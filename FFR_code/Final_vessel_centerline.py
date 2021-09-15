import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def nothing(x):
    pass

def skeletonize(img):
    gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = np.size(gray_img)
    skel = np.zeros(gray_img.shape,np.uint8)

    ret,img = cv2.threshold(gray_img,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False

    while(not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    
    return skel
    #return corner(cv2.cvtColor(skel,cv2.COLOR_GRAY2BGR))
    
def corner(img):

    # cornerHarris : 코너 검출 알고리즘
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 5, 3, 0.05)
    
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]
    return img

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
    QCA_mask = cv2.inRange(img_hsv,(0,0,255),(0,0,255)) # white
    #QCA_mask = cv2.inRange(img_hsv,(0,255,255),(180,255,255)) # yellow
    #QCA_mask=QCA_crop[QCA_crop==(255,255,255)]
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

def contour_mask(img,mode):
    # img : white 추출된 image
    # mode : input이 Color인지 Gray인지
    if mode=="gray":
        gray=img.copy()
        img=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if mode=="color":
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res=np.zeros(img.shape,np.uint8)

    _,thresh=cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    L=[]

    for i in range(len(contours)):
        if(cv2.arcLength(contours[i], True))>200:
            L.append(contours[i])
            #print(cv2.contourArea(contours[i]),cv2.arcLength(contours[i], False))
            #cv2.drawContours(res, [contours[i]], 0, (0, 0, 255), -1)
            #cv2.putText(res, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
            #cv2.circle(res,tuple(contours[i][len(contours[i])//2][0]),3,(0,0,255),-1)
        
        #cv2.drawContours(res, [contours[i]], 0, (0, 0, 255), 1)

    # print("contour num : ", len(L))
    #print(L[0].shape,L[1].shape)
    #print(np.argmax(L[0],axis=0),np.argmax(L[1],axis=0))

    arr=np.concatenate((L[0][0:len(L[0])//2],L[1][len(L[1])//2-1::-1]), axis=0)
    cv2.drawContours(res, [arr], 0, (255, 255,255), -1)
    #print(arr.shape)
    
    res=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)

    return res

def contour_in_mask(res_img):
    # res_img : (512,512,3)

    res=np.zeros(res_img.shape,np.uint8)
    gray=cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _,thresh=cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
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

def detect_line(gray):
    img=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    res=np.zeros(img.shape,np.uint8)

    _,thresh=cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    L=[]

    max_length=0
    max_ind=0

    for i in range(len(contours)):
        length = cv2.arcLength(contours[i], True)
        if length>max_length:
            max_length = length
            max_ind = i

    # for i in range(len(contours)):
    #     if(cv2.arcLength(contours[i], True))>10:
    #         #L.append(contours[i])
    #         cv2.drawContours(res, [contours[i]], 0, (0, 0, 255), 1)

    cv2.drawContours(res, [contours[max_ind]], 0, (255, 255, 255), -1)
    #cv2.circle(res,tuple(contours[max_ind][len(contours[i])//2][0]),3,(255,255,255),-1)

    #return corner(res)
    return cv2.ximgproc.thinning(cv2.cvtColor(res,cv2.COLOR_BGR2GRAY))

def show(num):
    #yellow_res_mask=cv2.imread(res_mask_path[num])
    #cv2.imshow('yellow_res_mask',yellow_res_mask)
    #cv2.imshow('yellow_skeleton',skeletonize(yellow_res_mask))

    #cv2.imshow('corner',contour(res_mask))

    QCA_img=cv2.imread(imges_path[num][0])
    img_512=cv2.imread(imges_path[num][1])
    QCA_crop, QCA_white = make_mask(QCA_img,img_512)

    #cv2.imshow('QCA_crop',QCA_crop)
    #cv2.imshow('img_512',img_512)
    #cv2.imshow('QCA_white',QCA_white)

    QCA_contour_mask=contour_mask(QCA_white,"gray")
    #cv2.imshow('QCA_contour_mask', QCA_contour_mask)
    
    res_mask=move_mask(QCA_crop,img_512,QCA_contour_mask)
    cv2.imshow('res_mask', res_mask)

    res_img = cv2.bitwise_and(img_512,img_512,mask=res_mask) 
    cv2.imshow('res_img', res_img)

    # normalize=cv2.normalize(cv2.cvtColor(res_img,cv2.COLOR_BGR2GRAY), None, 0, 255, cv2.NORM_MINMAX,mask=res_mask)
    # cv2.imshow('normalize', normalize)
    # tresh=cv2.threshold(normalize,200, 255,cv2.THRESH_BINARY_INV)[1]
    # #cv2.imshow("thresh",tresh)
    # cv2.imshow("thresh",cv2.bitwise_and(tresh,tresh,mask=res_mask))

    # cv2.imshow("edges",cv2.Canny(cv2.cvtColor(res_img,cv2.COLOR_BGR2GRAY),100,200))
    
    # cv2.imshow('contour_in_mask', contour_in_mask(res_img))

    #cv2.imshow('corner',corner(cv2.cvtColor(res_mask,cv2.COLOR_GRAY2BGR)))
    
    #skeleton=skeletonize(cv2.cvtColor(res_mask,cv2.COLOR_GRAY2BGR))
    skeleton=cv2.ximgproc.thinning(res_mask)
    cv2.imshow('skeleton',skeleton)

    skeleton=detect_line(skeleton) # make one line
    cv2.imshow('skeleton_contour',skeleton)
    # if(not os.path.isdir(os.path.join('..',"generated data",patend_ids[num]))):
    #     os.mkdir(os.path.join('..',"generated data",patend_ids[num]))

    ### save generated data ##
    # cv2.imwrite(os.path.join('..',"generated data",patend_ids[num],"skeleton.png"),
    #            skeleton)
    # cv2.imwrite(os.path.join('..', "generated data",patend_ids[num],"QCA_mask" + ".png"),
    #            res_mask)

    skel_check = res_img.copy()
    skel_check[skeleton==255] = [0,0,255]
    cv2.imshow('skel_check',skel_check)

    # print(res_img.shape)
    # hist = cv2.calcHist([cv2.cvtColor(res_img,cv2.COLOR_BGR2GRAY)], [0], res_mask, [256], [0,256])
    # plt.plot(hist)
    # plt.show()

num=0

patend_ids=os.listdir('../2nd data')
imges_path = [(os.path.join('..','2nd data',i,'AP','AP_QCA.bmp'),os.path.join('..','2nd data',i,'AP','AP_contrast.bmp')) for i in patend_ids]
#res_mask_path=[os.path.join('..',"generated data","QCA_extract_mask",i) for i in os.listdir(os.path.join('..',"generated data","QCA_extract_mask"))]

show(num)

remove_window_name =('res_mask','skeleton','check_res')

while(1):
    key = cv2.waitKeyEx(1)

    if  key == 27:
        break
    elif key==122:
        # for i in remove_window_name:
        #     cv2.destroyWindow(i)
        num = num-1 if num-1>0 else 0
        print(num,patend_ids[num])

        show(num)

    elif key==120:
        # for i in remove_window_name:
        #     cv2.destroyWindow(i)
        num+=1
        assert 0<=num<len(patend_ids)
        print(num,patend_ids[num])

        show(num)
    
    elif key==32:
        pass
        # if(os.path.isdir(os.path.join('..',"generated data","QCA_extract_mask"))):
        #     cv2.imwrite(os.path.join('..',"generated data","QCA_extract_mask",patend_ids[num]+"_QCA_mask"+".png"),res_mask)
        #     print(num,patend_ids[num],"is saved!")


