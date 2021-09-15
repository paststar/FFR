import os
import cv2
import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
import seaborn as sns

patient_ids=os.listdir('../2nd data')
patient_ids.remove('1891170') # no pull back graph
patient_ids.remove('739092') # no pull back graph

images_path = [(os.path.join('..','2nd data',i,'AP','AP_QCA.bmp'), os.path.join('..','2nd data',i,'AP','AP_contrast.bmp'), os.path.join('..','2nd data',i,'AP','AP_pre.bmp')) for i in patient_ids]
reg_images_path = [os.path.join('..',"generated data",i,"nongrid_reg.png") for i in patient_ids]

# vessel_mask_path=[os.path.join('..',"generated data",i,"superpixels_intersection.png") for i in patient_ids]
vessel_mask_path = [os.path.join('..',"generated data",i,"active_contour_mask.png") for i in patient_ids]
# vessel_mask_path=[os.path.join('..',"generated data",i,"QCA_mask.png") for i in patient_ids]


pre_img_mean = np.array([])
img_mean = np.array([])
reg_mean = np.array([])
mask_mean = np.array([])
mask_std = np.array([])

for i in range(len(patient_ids)):
    #print("patient_ids :",patient_ids[i])

    pre_img = cv2.imread(images_path[i][2], 0)
    img = cv2.imread(images_path[i][1],0)
    # reg_img = cv2.imread(reg_images_path[i],0)

    vessel_mask = cv2.imread(vessel_mask_path[i],0)
    res = ma.masked_array(img, mask=~vessel_mask.astype(bool))
    mask_std = np.append(mask_std, np.std(res))

    #res = cv2.bitwise_and(img,img,mask=vessel_mask)
    #mask_mean = np.append(mask_mean,np.sum(res)/np.sum(vessel_mask==255))

    pre_img_mean = np.append(pre_img_mean,np.mean(pre_img))
    # reg_mean = np.append(reg_mean,np.mean(reg_img))
    img_mean = np.append(img_mean,np.mean(img))

#print(pre_img_mean)
#print(img_mean)

#plt.hist(pre_img_mean, bins=30, density=True, alpha=0.7, histtype='step')
#plt.hist(img_mean, bins=30, density=True, alpha=0.9, histtype='step')

sns.displot(pre_img_mean,bins=50)
sns.displot(img_mean,bins=50)
#sns.displot(reg_mean,bins=70)
#sns.displot(mask_std,bins=20)
plt.show()



