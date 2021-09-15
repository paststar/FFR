import os
import cv2
import numpy as np

def nothing(x):
    pass

def show(num):
    id = patient_ids[num]
    print("patient id :",id)

    reg = cv2.imread(reg_images_path[num],0)
    #before_reg = cv2.imread(before_reg_path[num],0)
    vessel_mask = cv2.imread(vessel_mask_path[num], 0)
    graph = cv2.imread(graph_path[num])
    img = cv2.imread(imges_path[num][1],0)

    roi = cv2.bitwise_and(reg, reg, mask=vessel_mask)

    cv2.imshow("reg", reg)
    cv2.imshow("roi", roi)
    cv2.imshow("graph", graph)

    cv2.imshow("pseudo_img",cv2.bitwise_and(cv2.applyColorMap(img,cv2.COLORMAP_JET),
                                     cv2.applyColorMap(img,cv2.COLORMAP_JET),
                                     mask = vessel_mask))

    #cv2.imshow("before_reg", cv2.applyColorMap(before_reg, cv2.COLORMAP_JET))

    cv2.imshow("pseudo_roi",cv2.bitwise_and(cv2.applyColorMap(roi,cv2.COLORMAP_JET),
                                     cv2.applyColorMap(roi,cv2.COLORMAP_JET),
                                     mask=vessel_mask))
    
    return vessel_mask, roi

patient_ids = os.listdir('../2nd data')
patient_ids.remove('1891170')  # no pull back
patient_ids.remove('739092')  # no pull back
#patient_ids.remove('1908822') # bad translation

#reg_images_path = [os.path.join('..',"generated data",i,"multimodal_translation_reg.png") for i in patient_ids]
#reg_images_path = [os.path.join('..',"generated data",i,"nongrid2_reg.png") for i in patient_ids]
reg_images_path = [os.path.join('..',"generated data",i,"bottom_hat.png") for i in patient_ids]

before_reg_path = [os.path.join('..','2nd data',i,'AP','AP_reg.png') for i in patient_ids]
vessel_mask_path=[os.path.join('..',"generated data",i,"active_contour_mask.png") for i in patient_ids]
graph_path = [os.path.join('..','2nd data',i,'FFR_pullback.jpg') for i in patient_ids]
imges_path = [(os.path.join('..','2nd data',i,'AP','AP_QCA.bmp'),os.path.join('..','2nd data',i,'AP','AP_contrast.bmp')) for i in patient_ids]

num = 0
vessel_mask, roi = show(num)

cv2.namedWindow('track_bar')
cv2.createTrackbar('kernel size', 'track_bar', 1, 100, nothing)
cv2.createTrackbar('ratio', 'track_bar', 1, 100, nothing)

while (1):
    key = cv2.waitKeyEx(1)

    if key == 27:
        break

    elif key == 122:
        num = num - 1 if num - 1 > 0 else 0
        print(num, patient_ids[num])
        vessel_mask, roi = show(num)
        key = 32

    elif key == 120:
        num = num + 1 if num + 1 < len(patient_ids) else len(patient_ids) - 1
        print(num, patient_ids[num])
        vessel_mask, roi = show(num)
        key = 32

    if key == 32:
        kernel_size = cv2.getTrackbarPos('kernel size', 'track_bar')
        ratio = cv2.getTrackbarPos('ratio', 'track_bar') / 1000
        lsc = cv2.ximgproc.createSuperpixelLSC(roi, kernel_size, ratio)
        # lsc = cv2.ximgproc.createSuperpixelSLIC(roi,101,region_size=kernel_size,ruler=ratio*10000)
        lsc.iterate(10)

        mask_lsc = lsc.getLabelContourMask()
        mask_inv_lsc = cv2.bitwise_not(mask_lsc)
        img_lsc = cv2.cvtColor(roi,cv2.COLOR_GRAY2BGR)
        img_lsc[mask_lsc == 255] = [0, 0, 255]
        # img_lsc = cv2.bitwise_and(roi,roi,mask = mask_inv_lsc)
        cv2.imshow("super_pixel",img_lsc)

        number_lsc = lsc.getNumberOfSuperpixels()
        label_lsc = lsc.getLabels()
        # contour_label_lsc = lsc.getLabelContourMask() #??
        # print(label_lsc,number_lsc)

        superpixel_mean = np.zeros(img_lsc.shape,dtype=np.uint8)
        for i in range(number_lsc):
            superpixel_mean[label_lsc==i] = np.mean(roi[label_lsc == i])

        cv2.imshow("mean", superpixel_mean)
        cv2.imshow("mean_pseudo_coloring",cv2.bitwise_and(cv2.applyColorMap(superpixel_mean,cv2.COLORMAP_JET), cv2.applyColorMap(superpixel_mean,cv2.COLORMAP_JET), mask=vessel_mask))

    if key == 9:
        #cv2.imwrite(os.path.join("../generated data", patient_ids[num], 'bottom_hat_activecontour_superpixel_mean.png'),superpixel_mean)
        print(patient_ids[num],"is saved!!!")
