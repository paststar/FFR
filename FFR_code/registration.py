import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.data import stereo_motorcycle, vortex
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1, optical_flow_ilk

def nothing(x):
    pass

patient_ids=os.listdir('../2nd data')
patient_ids.remove('1891170') # no pull back graph
patient_ids.remove('739092') # no pull back graph

def show(id):

    img = cv2.imread(os.path.join('..', '2nd data', id, 'AP', 'AP_contrast.bmp'), 0)
    pre_img = cv2.imread(os.path.join('..', '2nd data', id, 'AP', 'AP_pre.bmp'), 0)
    #QCA_mask = cv2.imread(os.path.join('..', "generated data", id, "QCA_mask.png"), 0)

    cv2.imshow("img",img)
    cv2.imshow("pre_img",pre_img)
    cv2.imshow("subtract",cv2.applyColorMap(cv2.subtract(img,pre_img),cv2.COLORMAP_JET))
    return img, pre_img

num=0

cv2.namedWindow('track_bar')
#cv2.createTrackbar('alpha','track_bar',1,100,nothing)
#cv2.createTrackbar('beta','track_bar',1,100,nothing)
#cv2.createTrackbar('gamma','track_bar',1,200,nothing)

img, pre_img = show(patient_ids[num])

while(1):
    key = cv2.waitKeyEx(1)

    if key == 27:
        break

    elif key == 122:
        # for i in remove_window_name:
        #     cv2.destroyWindow(i)
        num = num - 1 if num - 1 > 0 else 0
        print(num, patient_ids[num])

        img, pre_img = show(patient_ids[num])

        key = 32

    elif key == 120:
        # for i in remove_window_name:
        #     cv2.destroyWindow(i)
        num += 1
        num = num + 1 if num + 1 < len(patient_ids) else len(patient_ids)-1

        #assert 0 <= num < len(patient_ids)
        print(num, patient_ids[num])

        img, pre_img = show(patient_ids[num])

        key = 32

    if key == 32:
        # sift
        sift = cv2.xfeatures2d.SIFT_create()

        keypoints_1, descriptors_1 = sift.detectAndCompute(img, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(pre_img, None)

        # feature matching
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

        matches = bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x: x.distance)

        img3 = cv2.drawMatches(img, keypoints_1, pre_img, keypoints_2, matches[:50], pre_img, flags=2)
        cv2.imshow("res",img3)


        # warp_mode = cv2.MOTION_AFFINE
        #
        # if warp_mode == cv2.MOTION_HOMOGRAPHY:
        #     warp_matrix = np.eye(3, 3, dtype=np.float32)
        # else:
        #     warp_matrix = np.eye(2, 3, dtype=np.float32)
        #
        # number_of_iterations = 1000;
        # termination_eps = 1e-10;
        #
        # tmp = img.copy()
        # tmp[img<90] = np.mean(pre_img)
        #
        # cv2.imshow("tmp",tmp)
        # criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
        # (cc, warp_matrix) = cv2.findTransformECC(tmp,pre_img, warp_matrix, warp_mode, criteria)
        #
        # if warp_mode == cv2.MOTION_HOMOGRAPHY:
        #     img_aligned = cv2.warpPerspective(pre_img, warp_matrix, img.shape,
        #                                   flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        # else:
        #     img_aligned = cv2.warpAffine(pre_img, warp_matrix, img.shape, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        #
        # cv2.imshow("img_aligned",img_aligned)
        # cv2.imshow("image_warp_subtract", cv2.applyColorMap(cv2.subtract(img, img_aligned), cv2.COLORMAP_JET))

    if key == 115:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")



