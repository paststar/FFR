import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.segmentation import active_contour
import cv2

def nothing(x):
    pass

patient_ids=os.listdir('../2nd data')
patient_ids.remove('1891170') # no pull back graph
patient_ids.remove('739092') # no pull back graph

def show(id):
    img = cv2.imread(os.path.join('..', '2nd data', id, 'AP', 'AP_contrast.bmp'), 0)
    #img = cv2.imread(os.path.join('..', '2nd data', id, 'AP', 'AP_reg.png'), 0)
    QCA_mask = cv2.imread(os.path.join('..', "generated data", id, "QCA_mask.png"), 0)
    before_skeleton = cv2.imread(os.path.join('..', "generated data", id, "skeleton.png"), 0)

    return img, QCA_mask, before_skeleton

num=0

cv2.namedWindow('track_bar')
cv2.createTrackbar('alpha','track_bar',1,100,nothing)
cv2.createTrackbar('beta','track_bar',1,100,nothing)
cv2.createTrackbar('gamma','track_bar',1,200,nothing)

img_512, QCA_mask, before_skeleton = show(patient_ids[num])

while(1):
    key = cv2.waitKeyEx(1)

    if key == 27:
        break

    elif key == 122:
        # for i in remove_window_name:
        #     cv2.destroyWindow(i)
        num = num - 1 if num - 1 > 0 else 0
        print(num, patient_ids[num])

        img_512, QCA_mask, before_skeleton = show(patient_ids[num])

        key = 32

    elif key == 120:
        # for i in remove_window_name:
        #     cv2.destroyWindow(i)
        #num += 1
        num = num +1 if num +1 < len(patient_ids)  else len(patient_ids)-1

        #assert 0 <= num < len(patient_ids)
        print(num, patient_ids[num])

        img_512, QCA_mask, before_skeleton = show(patient_ids[num])

        key = 32

    if key == 32:
        alpha = cv2.getTrackbarPos('alpha','track_bar')/10000 # Continuity
        beta = cv2.getTrackbarPos('beta','track_bar')/10 # Curvature
        gamma = cv2.getTrackbarPos('gamma','track_bar')/1000 # Gradient

        _, thresh = cv2.threshold(QCA_mask, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        contour_length = np.array([])
        for j in range(len(contours)):
            contour_length = np.append(contour_length, cv2.arcLength(contours[j], False))

        init = np.flip(np.squeeze(contours[np.argmax(contour_length)], axis=1), axis=1)
        snake = active_contour(img_512, init, alpha=alpha, beta=beta, gamma=gamma, coordinates='rc')

        tmp = img_512.copy()
        tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(tmp, [contours[np.argmax(contour_length)]], 0, (255, 0, 0), 1)
        cv2.drawContours(tmp, [np.flip(snake.astype(np.int64), axis=1)], 0, (0, 0, 255), 1)
        cv2.imshow("active_contour_result", tmp)
        print(f"alpha : {alpha} beta : {beta} gamma : {gamma} ")

        active_contour_mask = np.zeros(img_512.shape,dtype=np.uint8)
        cv2.drawContours(active_contour_mask, [np.flip(snake.astype(np.int64), axis=1)], 0, (255, 255, 255), -1)
        skeleton = cv2.ximgproc.thinning(active_contour_mask)
        cv2.imshow("skeleton", skeleton)

        tmp = cv2.cvtColor(active_contour_mask, cv2.COLOR_GRAY2BGR)
        tmp[skeleton==255] = [0,0,255]
        cv2.imshow("active_contour_mask", tmp)

        tmp = cv2.cvtColor(img_512, cv2.COLOR_GRAY2BGR)
        tmp[before_skeleton == 255] = [255, 0, 0]
        tmp[skeleton == 255] = [0, 0, 255]
        cv2.imshow("res_skeleton",tmp)

    if key == 9:
        # hyper parameter = 7, 31, 4
        print(f"{patient_ids[num]} is saved!!!")
        cv2.imwrite(os.path.join('..', "generated data", patient_ids[num], "active_contour_mask.png"), active_contour_mask)
        cv2.imwrite(os.path.join('..', "generated data", patient_ids[num], "active_contour_skeleton.png"),skeleton)
        #cv2.waitKey(0)


