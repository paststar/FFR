import cv2
import numpy as np
import os

patend_ids=['1538839','1745659', '1891170', '1892844']
#patend_ids=os.listdir('2nd data')
imges_path = [(os.path.join('2nd data',i,'AP','AP_QCA.bmp'),os.path.join('2nd data',i,'AP','AP_contrast.bmp')) for i in patend_ids]


for i in range(len(patend_ids)):
    QCA_img=cv2.imread(imges_path[i][0])

    crop_range=[]
    for j in range(QCA_img.shape[1]-1):
        if bool(QCA_img[:581,j,:].sum())^bool(QCA_img[:581,j+1,:].sum()):
            crop_range.append(j+1)
    print(i,patend_ids[i],crop_range)
    if len(crop_range)!=2:
        #print("BUG! BUG! BUG! BUG! BUG! BUG! BUG! BUG!")
        crop_range=[84,810]
    cv2.imshow('QCA_crop',QCA_img[:581,crop_range[0]:crop_range[1],:])
    #print(QCA_img[:581,crop_range[0]:crop_range[1],:].shape )

    cv2.waitKey()



    
