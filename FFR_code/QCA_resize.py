import cv2
import numpy as np

QCA_img=cv2.imread('2nd data\\29563\\AP\\AP_QCA.bmp')
#QCA_img=cv2.resize(QCA_img[:581,84:810,:],(512,512))
QCA_img=cv2.resize(QCA_img[:581,84:810,:],dsize=(0, 0), fx=512/726, fy=512/726)
img=cv2.imread('2nd data\\29563\\AP\\AP_contrast.bmp')

im1_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(QCA_img,cv2.COLOR_BGR2GRAY)

sz = img.shape
warp_mode = cv2.MOTION_TRANSLATION
number_of_iterations = 5000
termination_eps = 1e-10
warp_matrix = np.eye(2, 3, dtype=np.float32)
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria,inputMask=None,gaussFiltSize=1)
im2_aligned = cv2.warpAffine(QCA_img, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

im2_aligned=cv2.circle(im2_aligned,(100,100),10,(255,0,0),-1)
img=cv2.circle(img,(100,100),10,(255,0,0),-1)
QCA_img=cv2.circle(QCA_img,(100,100),10,(255,0,0),-1)

print(warp_matrix)
cv2.imshow("img", img)
cv2.imshow("QCA_img", QCA_img)
cv2.imshow("Aligned Image 2", im2_aligned)
cv2.waitKey()