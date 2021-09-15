import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.interpolate import interp1d
import pickle
from sklearn.metrics import mutual_info_score

patient_ids=os.listdir('../2nd data')
patient_ids.remove('1891170') # no pull back graph
patient_ids.remove('739092') # no pull back graph
patient_ids.remove('415865') # no reg.png (registration)

skeleton_path=[os.path.join('..',"generated data",i,"skeleton.png") for i in patient_ids]
vessel_mask_path=[os.path.join('..',"generated data",i,"superpixels_intersection.png") for i in patient_ids]
images_path = [(os.path.join('..','2nd data',i,'AP','AP_QCA.bmp'),os.path.join('..','2nd data',i,'AP','AP_contrast.bmp')) for i in patient_ids]
reg_images_path = [os.path.join('..','2nd data',i,'AP','AP_reg.png') for i in patient_ids]
graph_path = [os.path.join('..','2nd data',i,'FFR_pullback.jpg') for i in patient_ids]
QCA_mask_path = [os.path.join('..','generated data',i,'QCA_mask.png') for i in patient_ids]

for i in range(len(patient_ids)):
    print("num : ",i,"patient_ids :",patient_ids[i])
    img_512 = cv2.imread(images_path[i][1],0)
    reg_img = cv2.imread(reg_images_path[i], 0)
    QCA_mask = cv2.imread(QCA_mask_path[i], 0)
    skeleton = cv2.imread(skeleton_path[i], 0)
    vessel_mask = cv2.imread(vessel_mask_path[i], 0)

    print(reg_img.shape)
    reg_img = cv2.resize(reg_img,(512,512))


    ### GT graph ###
    graph_img = cv2.imread(graph_path[i])
    #[150:,:870,:]
    cv2.imshow("graph_img", graph_img)

    #cv2.imshow("img_512", img_512)
    cv2.imshow("reg_img", reg_img)
    #cv2.imshow("skeleton", skeleton)
    #cv2.imshow("QCA_mask", QCA_mask)
    #cv2.imshow("vessel_mask", vessel_mask)

    #tmp = cv2.cvtColor(img_512,cv2.COLOR_GRAY2BGR)
    #tmp[skeleton==255] = [0,0,255]
    #tmp = cv2.circle(tmp, (268, 135), 3, [255, 0, 0], -1)
    tmp = cv2.bitwise_and(img_512,img_512, mask=QCA_mask)
    cv2.imshow("res", tmp)

    #tmp = cv2.cvtColor(reg_img, cv2.COLOR_GRAY2BGR)
    #tmp[skeleton == 255] = [0, 0, 255]
    #cv2.imshow("reg_img",tmp)

    #tmp = cv2.circle(tmp, (268, 135), 3, [255, 0, 0], -1)
    #tmp = cv2.bitwise_and(tmp, tmp, mask=QCA_mask)
    #cv2.imshow("reg_mask_img", tmp)

    #cv2.imshow("mask_subtract", QCA_mask - vessel_mask)

    cv2.waitKey(0)