import cv2
import numpy as np
import os


def nothing(x):
    pass

def vessel_mask_contour(vessel_mask):
    #vessel_mask = cv2.imread(vessel_mask_path[i], 0)
    _, thresh = cv2.threshold(vessel_mask, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    bool_contour = np.zeros(vessel_mask.shape, dtype=np.bool)

    contour_length = np.array([])
    for j in range(len(contours)):
        contour_length = np.append(contour_length, cv2.arcLength(contours[j], False))
        # print(cv2.contourArea(contours[i]),cv2.arcLength(contours[i], False))
    # cv2.drawContours(tmp, [contours[np.argmax(contour_length)]], 0, (0, 0, 255), 1)

    for j in contours[np.argmax(contour_length)]:
        bool_contour[j[0][1]][j[0][0]] = True

    return bool_contour

def show(num):
    img = cv2.imread(images_path[num][1],0)
    #img = cv2.resize(cv2.imread(images_path[num][1], 0), dsize=(0, 0), fx=0.9, fy=0.9)

    before_reg = cv2.resize(cv2.imread(before_reg_path[num], 0),(512,512))
    #before_reg = cv2.resize(cv2.resize(cv2.imread(before_reg_path[num], 0), (512, 512), 0), dsize=(0, 0), fx=0.9, fy=0.9)

    reg = cv2.imread(reg_images_path[num], 0)
    #reg = cv2.resize(cv2.imread(reg_images_path[num], 0), dsize=(0, 0), fx=0.9,fy=0.9)

    ### vessel_mask_contour ###

    D = {"img" : img, "before_reg" : before_reg, "reg" : reg}

    COLOR = {
        0 : [255,0,0],
        1 : [0, 255, 0],
        2 : [0, 0, 255]
    }

    # for j in D.keys():
    #     tmp = cv2.cvtColor(D[j],cv2.COLOR_GRAY2BGR)
    #     for i in range(3):
    #         vessel_mask = cv2.resize(cv2.imread(vessel_mask_path[num][i], 0), dsize=(0, 0), fx=0.9, fy=0.9)
    #         bool_contour = vessel_mask_contour(vessel_mask)
    #         tmp[bool_contour] = COLOR[i]
    #
    #     cv2.imshow(j,tmp)

        # cv2.imshow("img_"+str(i), img)
        # cv2.imshow("before_reg_"+str(i), before_reg)
        # cv2.imshow("reg_"+str(i), reg)

    ### pseudo coloring ###
    cv2.imshow("pseudo_img", cv2.applyColorMap(img, cv2.COLORMAP_JET))
    cv2.imshow("pseudo_before_reg", cv2.applyColorMap(before_reg, cv2.COLORMAP_JET))
    cv2.imshow("pseudo_reg", cv2.applyColorMap(reg, cv2.COLORMAP_JET))

    print(np.unique(reg))
    print(np.unique(before_reg))

    ### graph ###
    graph = cv2.resize(cv2.imread(graph_path[num]), dsize=(0,0), fx=0.8, fy=0.8)
    cv2.imshow("graph", graph)

    print(np.mean(cv2.imread(images_path[num][1], 0)),np.mean(cv2.imread(images_path[num][2], 0)))


num = 0
patient_ids = os.listdir('../2nd data')
patient_ids.remove('1891170')  # no pull back
patient_ids.remove('739092')  # no pull back
patient_ids.remove('415865') # no reg.png (registration)

images_path = [(os.path.join('..','2nd data',i,'AP','AP_QCA.bmp'),os.path.join('..','2nd data',i,'AP','AP_contrast.bmp'), os.path.join('..','2nd data',i,'AP','AP_pre.bmp')) for i in patient_ids]
before_reg_path = [os.path.join('..','2nd data',i,'AP','AP_reg.png') for i in patient_ids]
#reg_images_path = [os.path.join('..',"generated data",i,"nongrid2_reg.png") for i in patient_ids]
#reg_images_path = [os.path.join('..',"generated data",i,"multimodal_translation_reg.png") for i in patient_ids]
reg_images_path = [os.path.join('..',"generated data",i,"bottom_hat.png") for i in patient_ids]


graph_path = [os.path.join('..','2nd data',i,'FFR_pullback.jpg') for i in patient_ids]

vessel_mask_path=[(os.path.join('..',"generated data",i,"superpixels_intersection.png"), os.path.join('..',"generated data",i,"active_contour_mask.png"), os.path.join('..',"generated data",i,"QCA_mask.png"))for i in patient_ids]
#vessel_mask_path=[os.path.join('..',"generated data",i,"active_contour_mask.png") for i in patient_ids]
#vessel_mask_path=[os.path.join('..',"generated data",i,"QCA_mask.png") for i in patient_ids]

print(reg_images_path)

show(num)
print(num, patient_ids[num])

# cv2.namedWindow('track_bar')
# cv2.createTrackbar('kernel size', 'track_bar', 1, 100, nothing)
# cv2.createTrackbar('ratio', 'track_bar', 1, 100, nothing)

remove_window_name = ('fill_mask', 'res_img', 'check_res', 'QCA_crop_img', 'img_512', 'pseudo coloring', 'skeleton')

while (1):
    key = cv2.waitKeyEx(1)

    if key == 27:
        break

    elif key == 122:
        num = num - 1 if num - 1 > 0 else 0
        print(num, patient_ids[num])
        show(num)
        #key = 32

    elif key == 120:

        num = num + 1 if num + 1 < len(patient_ids) else len(patient_ids) - 1
        print(num, patient_ids[num])
        show(num)
        #key = 32

    if key == 32:
        pass
