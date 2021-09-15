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

# skeleton_path=[os.path.join('..',"generated data",i,"skeleton.png") for i in patient_ids]
# vessel_mask_path=[os.path.join('..',"generated data",i,"superpixels_intersection.png") for i in patient_ids]
# images_path = [(os.path.join('..','2nd data',i,'AP','AP_QCA.bmp'),os.path.join('..','2nd data',i,'AP','AP_contrast.bmp')) for i in patient_ids]
# reg_images_path = [os.path.join('..','2nd data',i,'AP','AP_reg.png') for i in patient_ids]
graph_path = [os.path.join('..','2nd data',i,'FFR_pullback.jpg') for i in patient_ids]


for ind, id in enumerate(patient_ids):
    print(id)
    with open(os.path.join("../generated data",id, 'FFR_pullback.pickle'), 'rb') as fr:
         D = pickle.load(fr)
         true_x = D['X']
         true_y = D['Y']

    with open(os.path.join("../generated data", id, 'Center_contrast.pickle'), 'rb') as fr:
        D = pickle.load(fr)
        X = D['X']
        Y = D['Y']

    graph = cv2.imread(graph_path[ind])
    cv2.imshow("graph", graph)
    plt.plot(true_x,true_y,'-',X,Y/max(Y),'--')
    plt.show()
