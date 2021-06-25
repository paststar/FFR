import cv2
import numpy as np
import os

from numpy.lib.arraysetops import unique

def nothing(x):
    pass

def show(num):
    # QCA_img=cv2.imread(imges_path[num][0])
    # img_512=cv2.imread(imges_path[num][1])
    # skeleton=cv2.imread(skeleton_path[num],0)
    QCA_img = cv2.imread((os.path.join('..','2nd data',patend_ids[num],'AP','AP_QCA.bmp')))
    img_512 = cv2.imread((os.path.join('..','2nd data',patend_ids[num],'AP','AP_contrast.bmp')))
    skeleton = cv2.imread(os.path.join('..',"generated data",patend_ids[num],"skeleton.png"),0)

    QCA_crop,_ = make_mask(QCA_img,img_512)
        
    #cv2.imshow('QCA_crop_img',QCA_crop)
    #cv2.imshow('img_512',img_512)
    #cv2.imshow('skeleton',skeleton)

    pullback = cv2.imread(os.path.join('..', "2nd data", patend_ids[num], "FFR_pullback.jpg"))[150:,:880,:]
    pullback = cv2.resize(pullback,dsize=(0, 0),fx=0.8, fy=0.8)
    cv2.imshow('pullback',pullback)

    return QCA_crop,img_512,skeleton

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
patend_ids=os.listdir('../2nd data')
patend_ids.remove('1891170') # no pull back
patend_ids.remove('739092') # no pull back

#imges_path = [(os.path.join('..','2nd data',i,'AP','AP_QCA.bmp'),os.path.join('..','2nd data',i,'AP','AP_contrast.bmp')) for i in patend_ids]
#skeleton_path=[os.path.join('..',"generated data","skeleton",i) for i in os.listdir(os.path.join('..',"generated data","skeleton"))]


QCA_crop,img_512,skeleton=show(num)
print(num,patend_ids[num])

cv2.namedWindow('track_bar')
cv2.createTrackbar('kernel size','track_bar',1,100,nothing)
cv2.createTrackbar('ratio','track_bar',1,100,nothing)

remove_window_name =('fill_mask','res_img','check_res','QCA_crop_img','img_512','pseudo coloring','skeleton')

while(1):
    key = cv2.waitKeyEx(1)
    
    if  key == 27:
        break
    elif key==122:
        # for i in remove_window_name:
        #     cv2.destroyWindow(i)
        num = num-1 if num-1>0 else 0
        print(num,patend_ids[num])
        
        QCA_crop,img_512,skeleton=show(num)

        key=32
        
    elif key==120:
        # for i in remove_window_name:
        #     cv2.destroyWindow(i)
        num+=1
        assert 0<=num<len(patend_ids)
        print(num,patend_ids[num])
        
        QCA_crop,img_512,skeleton=show(num)

        key=32
        

    if key==32:
        kernel_size = cv2.getTrackbarPos('kernel size','track_bar')
        ratio = cv2.getTrackbarPos('ratio','track_bar')/1000

        lsc = cv2.ximgproc.createSuperpixelLSC(img_512,kernel_size,ratio)
        #lsc = cv2.ximgproc.createSuperpixelSLIC(img_512,101,region_size=kernel_size,ruler=ratio*10000)
        lsc.iterate(10)

        mask_lsc = lsc.getLabelContourMask()
        mask_inv_lsc = cv2.bitwise_not(mask_lsc)
        img_lsc = img_512.copy()
        img_lsc[mask_lsc==255]=[0,0,255]
        #img_lsc = cv2.bitwise_and(img_512,img_512,mask = mask_inv_lsc)
        # cv2.imshow("super_pixel",img_lsc)

        number_lsc = lsc.getNumberOfSuperpixels()
        label_lsc = lsc.getLabels()
        #contour_label_lsc = lsc.getLabelContourMask() #??
        #print(label_lsc,number_lsc)
        
        
        skeleton_intersection_superpixel = np.zeros(skeleton.shape,dtype=np.uint8)
        skeleton_intersection_superpixel_ind = np.unique(label_lsc[skeleton == 255])
        for i in skeleton_intersection_superpixel_ind:
            skeleton_intersection_superpixel[label_lsc==i] = 255

        check_skeleton_intersection_superpixel = cv2.cvtColor(skeleton_intersection_superpixel,cv2.COLOR_GRAY2BGR)
        check_skeleton_intersection_superpixel[skeleton==255] = [0,0,255]
        # cv2.imshow("skeleton_intersection_superpixel",check_skeleton_intersection_superpixel)
        check_skeleton_intersection_superpixel[mask_lsc==255]=[0,255,255]
        cv2.imshow("check_skeleton_intersection_superpixel",check_skeleton_intersection_superpixel)

        res_img = cv2.bitwise_and(img_512,img_512,mask=skeleton_intersection_superpixel)
        cv2.imshow('res_img', res_img)

        # tmp = img_512.copy()
        # tmp[skeleton==255] = [0,0,255]
        # cv2.imshow("tmp",tmp)

        superpixel_pseudo_mean = np.zeros(skeleton.shape,dtype=np.uint8)
        superpixel_pseudo_local_var = np.zeros(skeleton.shape,dtype=np.uint8)
        superpixel_pseudo_global_var = np.zeros(skeleton.shape, dtype=np.uint8)
        #superpixel_pseudo_global_std = np.zeros(skeleton.shape, dtype=np.uint8)

        superpixel_mean_array = np.array([])


        for i in range(number_lsc):
            superpixel_mean_array= np.append(superpixel_mean_array,np.mean(img_512[label_lsc == i]))
            superpixel_pseudo_mean[label_lsc==i] = superpixel_mean_array[i].astype(np.uint8)


        superpixel_var_array = (superpixel_mean_array - np.mean(superpixel_mean_array))**2
        #superpixel_std_array = np.abs(superpixel_mean_array - np.mean(superpixel_mean_array))
        #print(number_lsc, superpixel_var_array.shape)

        for i in range(number_lsc):
            superpixel_pseudo_local_var[label_lsc == i] = np.var(img_512[label_lsc == i]).astype(np.uint8)
            superpixel_pseudo_global_var[label_lsc == i] = superpixel_var_array[i].astype(np.uint8)
            #superpixel_pseudo_global_std[label_lsc == i] = superpixel_std_array[i].astype(np.uint8)

        #print(superpixel_pseudo_std)

        cv2.imshow("img_512_coloring",cv2.applyColorMap(cv2.cvtColor(img_512,cv2.COLOR_BGR2GRAY),cv2.COLORMAP_JET))
        superpixel_pseudo_mean = cv2.applyColorMap(superpixel_pseudo_mean, cv2.COLORMAP_JET)
        # superpixel_pseudo_mean[skeleton == 255] = [0,0,255]
        cv2.imshow("mean_coloring",superpixel_pseudo_mean)
        cv2.imshow("local_var_coloring",cv2.applyColorMap(superpixel_pseudo_local_var,cv2.COLORMAP_JET))
        #cv2.imshow("gloabl_var_coloring", cv2.applyColorMap(superpixel_pseudo_global_var, cv2.COLORMAP_JET))

        superpixel_pseudo_global_var = cv2.cvtColor(superpixel_pseudo_global_var, cv2.COLOR_GRAY2BGR)
        superpixel_pseudo_global_var[skeleton == 255] = [0,0,255]
        cv2.imshow("gloabl_var_coloring", superpixel_pseudo_global_var)

        ### ###
        superpixel_pseudo_intersection_var = np.ones(skeleton.shape, dtype=np.uint8)*255
        superpixel_intersection_mean_array = np.array([])
        for i in skeleton_intersection_superpixel_ind:
            superpixel_intersection_mean_array = np.append(superpixel_intersection_mean_array,np.mean(img_512[label_lsc == i]))

        superpixel_intersection_var_array = (superpixel_intersection_mean_array - np.mean(superpixel_intersection_mean_array))**2

        for i in range(len(skeleton_intersection_superpixel_ind)):
            superpixel_pseudo_intersection_var[label_lsc == skeleton_intersection_superpixel_ind[i]] = superpixel_intersection_var_array[i]

        cv2.imshow("intersectoin_gloabl_var_coloring", superpixel_pseudo_intersection_var)
