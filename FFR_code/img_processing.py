import cv2
import os
import pickle
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scipy.optimize import curve_fit #least square


from sklearn.feature_selection import  mutual_info_regression
from sklearn.linear_model import LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

patient_ids = os.listdir('../2nd data')
patient_ids.remove('1891170')  # no pull back
patient_ids.remove('739092')  # no pull back
# patient_ids.remove('1908822') # bad translation
# patient_ids.remove('785547')
# patient_ids.remove('415865') # no reg.png (registration)

images_path = [(os.path.join('..', '2nd data', i, 'AP', 'AP_QCA.bmp'),
                os.path.join('..', '2nd data', i, 'AP', 'AP_contrast.bmp'),
                os.path.join('..', '2nd data', i, 'AP', 'AP_pre.bmp')) for i in patient_ids]
before_reg_path = [os.path.join('..', '2nd data', i, 'AP', 'AP_reg.png') for i in patient_ids]
reg_images_path = [os.path.join('..',"generated data",i,"nongrid2_reg.png") for i in patient_ids]
# reg_images_path = [os.path.join('..',"generated data",i,"multimodal_translation_reg.png") for i in patient_ids]
# reg_images_path = [os.path.join('..', "generated data", i, "nongrid2_activecontour_superpixel_mean.png") for i in patient_ids]
# skeleton_path=[os.path.join('..',"generated data",i,"active_contour_skeleton.png") for i in patient_ids]

# graph_path = [os.path.join('..', '2nd data', i, 'FFR_pullback.jpg') for i in patient_ids]

vessel_mask_path = [(os.path.join('..', "generated data", i, "superpixels_intersection.png"),
                     os.path.join('..', "generated data", i, "active_contour_mask.png"),
                     os.path.join('..', "generated data", i, "QCA_mask.png")) for i in patient_ids]
# vessel_mask_path=[os.path.join('..',"generated data",i,"active_contour_mask.png") for i in patient_ids]
# vessel_mask_path=[os.path.join('..',"generated data",i,"QCA_mask.png") for i in patient_ids]

res_pickle = [
     'nongrid2_reg_active_contour_mask_centerline_density.pickle',
     'nongrid2_reg_QCA_mask_centerline_density.pickle',
    'nongrid2_activecontour_superpixel_mean_centerline.pickle',
     'multimodal_translation_active_contour_mask_centerline_density.pickle',
     'multimodal_translation_QCA_mask_centerline_density.pickle',
     'multimodal_translation_activecontour_superpixel_mean_centerline.pickle'
]


def show(name,img, color = False):
    if color:
        cv2.imshow(name,cv2.applyColorMap(img,cv2.COLORMAP_JET))
    else:
        cv2.imshow(name,img)


for i in range(len(patient_ids)):
    print(patient_ids[i])

    pre_img = cv2.imread(images_path[i][2],0)
    img = cv2.imread(images_path[i][1],0)
    show("pre_img", pre_img)
    show("img", img)

    ### divide ###
    # test = cv2.divide(img,pre_img)
    # show("test", test)
    # print(np.unique(test))

    filterSize = (17, 17)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,filterSize)

    ### closed ###
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    #show("opening",opening)
    show("closing", closing)

    # test = cv2.divide(closing,img)
    # show("test", test)
    # #print(np.unique(test))

    ### bottom-hat ###
    # res = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
    res = cv2.subtract(closing, img)
    show("res", res, False)
    show("pseudo_res",res,True)

    ### gaussian + bottom-hat ###
    closing_blur = cv2.GaussianBlur(closing, (7,7), 3)
    blur_res = cv2.subtract(closing_blur, img)
    show("closing_blur", closing_blur, False)
    show("blur_res", blur_res, False)
    show("pseudo_blur_res", blur_res, True)


    #cv2.imwrite(os.path.join("/home/bang/Desktop/bottom-hat", patient_ids[i] + "_bottom_hat.png"),res)

    ### background ###
    #show("background1", cv2.subtract(img, res), False)
    #show("background2",np.abs(img.astype(np.float_)-res.astype(np.float_)).astype(np.uint8), False)


    cv2.waitKey(0)