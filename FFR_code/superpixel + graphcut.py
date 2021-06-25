import cv2
import numpy as np
import os
import networkx as nx

def nothing(x):
    pass

def show(num):
    QCA_img=cv2.imread(imges_path[num][0])
    img_512=cv2.imread(imges_path[num][1])
    skeleton=cv2.imread(skeleton_path[num],0)

    QCA_crop,QCA_mask = make_mask(QCA_img,img_512)
        
    #cv2.imshow('QCA_crop_img',QCA_crop)
    #cv2.imshow('img_512',img_512)
    #cv2.imshow('skeleton',skeleton)

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

def superpixel2feature(img_512, label_lsc, target_num):
    # superpixel mean
    tmp = img_512[label_lsc == target_num]
    #print(tmp[:,0].sum() / tmp.shape[0])
    return tmp[:,0].sum() / tmp.shape[0]

def convolve2D(image, kernel, padding=1, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        #print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

def superpixel_adjacent(label_lsc,ind):
    kernel = np.array([[0,1,0], [1,0,1], [0,1,0]])
    mask = convolve2D(label_lsc==ind,kernel)>0
    res = np.unique(label_lsc[mask])
    return res
    
    

num=0
patend_ids=os.listdir('2nd data')
imges_path = [(os.path.join('2nd data',i,'AP','AP_QCA.bmp'),os.path.join('2nd data',i,'AP','AP_contrast.bmp')) for i in patend_ids]
skeleton_path=[os.path.join("generated data","skeleton",i) for i in os.listdir(os.path.join("generated data","skeleton"))]

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
    elif key==0x250000:
        # for i in remove_window_name:
        #     cv2.destroyWindow(i)
        num = num-1 if num-1>0 else 0
        print(num,patend_ids[num])
        
        QCA_crop,img_512,skeleton=show(num)

        key=32
        
    elif key==0x270000:
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

        number_lsc = lsc.getNumberOfSuperpixels()
        label_lsc = lsc.getLabels()
        contour_label_lsc = lsc.getLabelContourMask() # contour 부분 255로 표시

        img_lsc = img_512.copy()
        img_lsc[contour_label_lsc==255]=[0,0,255]
        cv2.imshow("super_pixel",img_lsc)
        #mask_inv_lsc = cv2.bitwise_not(contour_label_lsc)
        #img_lsc = cv2.bitwise_and(img_512,img_512,mask = mask_inv_lsc)
            
        graph = nx.Graph() # 무방향성 그래프
        n = np.max(label_lsc)+1 # node 개수
        #print(n)
        #print(np.unique(label_lsc))

        # target=120
        # adj=superpixel_adjacent(label_lsc,target)
        # adj = adj[adj>target]
        # tmp_img = img_512.copy()
        # tmp_img[label_lsc==target] = [255,0,0]
        # for i in adj:
        #     tmp_img[label_lsc==i] = [0,0,255]
        # cv2.imshow("tmp",tmp_img)

        for i in range(n):
            graph.add_node(i,val=superpixel2feature(img_512,label_lsc,i))

        for i in range(n):
            adj=superpixel_adjacent(label_lsc,i)
            adj = adj[adj>i]

            for j in adj:
                graph.add_edge(i,j,distance = 1)

        print(nx.classes.function.info(graph))



