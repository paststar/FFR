import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.interpolate import interp1d
import pickle
from scipy.optimize import curve_fit

patient_ids=os.listdir('../2nd data')
patient_ids.remove('1891170') # no pull back graph
patient_ids.remove('739092') # no pull back graph
#patient_ids.remove('1908822') # bad translation

#patient_ids.remove('415865') # no reg.png (registration)

#skeleton_path=[os.path.join('..',"generated data",i,"skeleton.png") for i in patient_ids]
skeleton_path=[os.path.join('..',"generated data",i,"active_contour_skeleton.png") for i in patient_ids]

#vessel_mask_path=[os.path.join('..',"generated data",i,"superpixels_intersection.png") for i in patient_ids]
#vessel_mask_path=[os.path.join('..',"generated data",i,"active_contour_mask.png") for i in patient_ids]
vessel_mask_path=[os.path.join('..',"generated data",i,"QCA_mask.png") for i in patient_ids]


images_path = [(os.path.join('..','2nd data',i,'AP','AP_QCA.bmp'),os.path.join('..','2nd data',i,'AP','AP_contrast.bmp'), os.path.join('..','2nd data',i,'AP','AP_pre.bmp')) for i in patient_ids]
#before_reg_path = [os.path.join('..','2nd data',i,'AP','AP_reg.png')for i in patient_ids]
#reg_images_path = [os.path.join('..',"generated data",i,"multimodal_translation_reg.png") for i in patient_ids]
#reg_images_path = [os.path.join('..',"generated data",i,"nongrid2_activecontour_superpixel_mean.png") for i in patient_ids]
reg_images_path = [os.path.join('..',"generated data",i,"bottom_hat.png") for i in patient_ids]

graph_path = [os.path.join('..','2nd data',i,'FFR_pullback.jpg') for i in patient_ids]

target = [
    ((-1,-1),(1,1)), # degree : 45
    ((-1,1),(1,-1)), # degree : 135
    ((-1,0),(1,0)), # degree : 0
    ((0,-1),(0,1)) # degree : 90
    ]

norm_grad = {
    1 : -1,
    2 : 1,
    3 : None,
    4 : 0
}

color_map = {
    1 : [255,0,0],
    2 : [0,255,0],
    3 : [0,0,255],
    4 : [0,255,255]
}

# clockwise
# d_x = [-1, -1, 0, 1, 1, 1, 0, -1]
# d_y = [0, 1, 1, 1, 0, -1, -1, -1]

d_x = [-1,0,1,0,-1,1,1,-1]
d_y = [0,1,0,-1,1,1,-1,-1]

# def cal_correlation(f, M,patient_id):
#     with open(os.path.join("../generated data",patient_id, 'FFR_pullback.pickle'), 'rb') as fr:
#         D = pickle.load(fr)
#     true_x = D['X']
#     true_y = D['Y']
#     xnew = np.linspace(0, M, num=len(true_x))
#     print(f"Correlation : {np.corrcoef(true_y,f(xnew))}")
#     print(f"Mutual Information : {mutual_info_score(true_y,f(xnew))}")

def traversal_visualize(skeleton, bool_contour):
    def check_range(tmp):
        if(0<=tmp and tmp<512):
            return True
        else:
            return False

    def cal_grad(x, y):
        for ind, d in enumerate(target):
            tmp_x1 = x + d[0][0]
            tmp_y1 = y + d[0][1]
            tmp_x2 = x + d[1][0]
            tmp_y2 = y + d[1][1]

            if(check_range(tmp_x1) and check_range(tmp_x2) and check_range(tmp_y1) and check_range(tmp_y2)):
                if(bool_skel[tmp_x1][tmp_y1] and bool_skel[tmp_x2][tmp_y2]):
                    #print(ind+1)
                    return ind+1
        return 0

    # skeleton : gray img
    bool_skel = skeleton == 255
    visited = bool_skel.copy()
    gradient_skel = np.zeros(skeleton.shape,dtype=np.uint8)
    #numbering_skel = np.zeros(skeleton.shape)
    cordinate_gradient_skel = []

    for i in range(visited.shape[0]):
        for j in range(visited.shape[1]):
            if(visited[i][j]):
                #print("start : ",i,j)
                visited[i][j] = False
                queue = deque([(i,j)])
                #num = 1

                while queue:
                    x,y = queue.popleft()
                    gradient_skel[x][y] = cal_grad(x,y)
                    if gradient_skel[x][y] != 0:
                        cordinate_gradient_skel.append((x,y))

                    ### ordering ###i
                    #numbering_skel[x][y] = num
                    #num += 1

                    for k in range(8):
                        tmp_x = x + d_x[k]
                        tmp_y = y + d_y[k]
                        if(not (check_range(tmp_x) and check_range(tmp_y))):
                            continue
                        if(visited[tmp_x][tmp_y]):
                            queue.append((tmp_x,tmp_y))
                            visited[tmp_x][tmp_y] = False

    ### remove outlier ###
    if len(cordinate_gradient_skel)>2:
        for i in range(1,len(cordinate_gradient_skel)-1):
            x0,y0 = cordinate_gradient_skel[i-1]
            x1,y1 = cordinate_gradient_skel[i]
            x2,y2 = cordinate_gradient_skel[i+1]

            if gradient_skel[x0][y0] == gradient_skel[x2][y2] :
                gradient_skel[x1][y1] = gradient_skel[x0][y0]
    #########################

    tmp = np.zeros(skeleton.shape,dtype=np.uint8)
    tmp = cv2.cvtColor(tmp,cv2.COLOR_GRAY2BGR)

    for i in range(visited.shape[0]):
        for j in range(visited.shape[1]):
            if(gradient_skel[i][j] != 0):
                a = norm_grad[gradient_skel[i][j]]
                color = color_map[gradient_skel[i][j]]
                if(a == None):
                    x = i
                    y = j
                    while (check_range(x) and check_range(y) and not bool_contour[x][y]):
                        tmp[x][y] = color
                        y = y + 1

                    y = j
                    while (check_range(x) and check_range(y) and not bool_contour[x][y]):
                        tmp[x][y] = color
                        x = x
                        y = y - 1

                elif(a==0):
                    x = i
                    y = j
                    while(check_range(x) and check_range(y) and not bool_contour[x][y]):
                        tmp[x][y] = color
                        x = x + 1

                    x = i
                    while (check_range(x) and check_range(y) and not bool_contour[x][y]):
                        tmp[x][y] = color
                        x = x - 1

                elif(a==1):
                    x = i
                    y = j
                    while (check_range(x) and check_range(y) and not bool_contour[x][y] and not bool_contour[x-1][y]*bool_contour[x][y-1]):
                        tmp[x][y] = color
                        x = x + 1
                        y = y + a * 1
                    x = i
                    y = j
                    while (check_range(x) and check_range(y) and not bool_contour[x][y] and not bool_contour[x+1][y]*bool_contour[x][y+1]):
                        tmp[x][y] = color
                        x = x - 1
                        y = y - a

                elif (a == -1):
                    x = i
                    y = j
                    while (check_range(x) and check_range(y) and not bool_contour[x][y] and not bool_contour[x-1][y]*bool_contour[x][y+1]):
                        tmp[x][y] = color
                        x = x + 1
                        y = y + a * 1
                    x = i
                    y = j
                    while (check_range(x) and check_range(y) and not bool_contour[x][y] and not bool_contour[x+1][y]*bool_contour[x][y-1]):
                        tmp[x][y] = color
                        x = x - 1
                        y = y - a

    tmp[bool_skel==True] = [255,255,255]
    tmp[bool_contour==True] = [255,255,255]
    cv2.imshow("norm_check", tmp)

    tmp = np.zeros(skeleton.shape, dtype=np.uint8)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
    tmp[gradient_skel == 1] = color_map[1]
    tmp[gradient_skel == 2] = color_map[2]
    tmp[gradient_skel == 3] = color_map[3]
    tmp[gradient_skel == 4] = color_map[4]
    cv2.imshow("gradient_skel_check", tmp)

def traversal(img_512, skeleton, bool_contour, patient_id):
    # skeleton : gray img
    def check_range(tmp):
        if(0<=tmp and tmp<512):
            return True
        else:
            return False

    bool_skel = skeleton == 255
    visited = bool_skel.copy()
    gradient_skel = np.zeros(skeleton.shape,dtype=np.uint8)
    numbering_skel = np.zeros(skeleton.shape)
    #cordinate_gradient_skel = []
    res = np.zeros(skeleton.shape,dtype=np.uint8)


    for i in range(visited.shape[0]):
        for j in range(visited.shape[1]):
            if(visited[i][j]):
                #print("start : ",i,j)
                visited[i][j] = False
                queue = deque([(i,j)])
                num = 1

                while queue:
                    x,y = queue.popleft()
                    res[x][y] = img_512[x][y]

                    ### ordering ###
                    numbering_skel[x][y] = num
                    num += 1

                    for k in range(8):
                        tmp_x = x + d_x[k]
                        tmp_y = y + d_y[k]
                        if(not (check_range(tmp_x) and check_range(tmp_y))):
                            continue

                        if(visited[tmp_x][tmp_y]):
                            queue.append((tmp_x,tmp_y))
                            visited[tmp_x][tmp_y] = False

    cv2.imshow("result", res)
    cv2.imshow("numbering_skel", numbering_skel.astype(np.uint8))

    X=[]
    Y=[]
    x = np.array([],dtype=np.int_)
    y = np.array([],dtype=np.int_)
    for i in range(1,num):
        t = np.where(numbering_skel == i)
        if(res[t] != 0):
            x = np.append(x, t[0])
            y = np.append(y, t[1])

    X.append(0)
    Y.append(res[x[0]][y[0]])
    for i in range(len(x)-1):
        X.append(X[i] + abs(x[i+1] - x[i]) + abs(y[i+1]-y[i]))
        Y.append(res[x[i+1]][y[i+1]])

    # def func(x, a, b, c, d, e):
    #     return a * (x**4) + b * (x**3) + c * (x**2) + d * (x**1) + e

    #print(len(X))
    X = X[-1] - np.array(X)
    #X = np.array(X)
    Y = np.array(Y)

    ### curve fit ###
    # X = X[10:-10]
    # Y = Y[10:-10]
    # Y = Y/ np.max(Y)
    #
    # def func(x, a, b, c, d, e, f, g):
    #     return a * (x ** 6) + b * (x ** 5) + c * (x ** 4) + d * (x ** 3) + e * (x ** 2) + f * x + g

    # popt, _ = curve_fit(func, X, Y)
    # x = np.linspace(0, max(X), num=max(X) * 10 + 1)
    # plt.xlim(min(X),max(X))
    # plt.ylim(0,1)
    #
    # plt.scatter(X, Y, marker='.')
    # plt.plot(x, func(x, *popt), color='red', linewidth=2)
    # plt.show(block=False)

    ################ Delete ######################
    # f = interp1d(X, Y, kind='cubic')
    # xnew = np.linspace(0, max(X), num=max(X) * 10 + 1)

    ### save ###
    res = {"X" : X, "Y" : Y}
    with open(os.path.join("../generated data", patient_id, 'bottom_hat_centerline.pickle'), 'wb') as fw:
        pickle.dump(res, fw)
        print(f"{patient_id} is saved!!!!")

    ### show ###
    # ynew = f(xnew)
    # ynew = ynew[::-1] / np.max(ynew)
    # plt.plot(xnew, ynew, '-',max(X)-X,Y/max(Y),'o')
    # #plt.show()

    #plt.show(block=False)

for i in range(len(patient_ids)):
    #i = i+1
    print("patient_ids :",patient_ids[i])
    #img_512 = cv2.imread(images_path[i][2], 0)
    #img_512 = cv2.subtract( cv2.imread(images_path[i][2],0),cv2.imread(images_path[i][1],0))
    img_512 = cv2.imread(reg_images_path[i], 0)

    skeleton = cv2.imread(skeleton_path[i],0)
    vessel_mask = cv2.imread(vessel_mask_path[i],0)
    graph_img = cv2.imread(graph_path[i])[150:,:870,:]

    cv2.imshow("img_512", img_512)
    #cv2.imshow("skeleton", skeleton)
    #cv2.imshow("QCA_mask", vessel_mask)
    cv2.imshow("graph_img", graph_img)

    _, thresh = cv2.threshold(vessel_mask, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    #print(len(contours))

    bool_contour = np.zeros(skeleton.shape,dtype=np.bool)

    contour_length = np.array([])
    for j in range(len(contours)):
        contour_length= np.append(contour_length,cv2.arcLength(contours[j], False))
        #print(cv2.contourArea(contours[i]),cv2.arcLength(contours[i], False))
    #cv2.drawContours(tmp, [contours[np.argmax(contour_length)]], 0, (0, 0, 255), 1)

    tmp = cv2.cvtColor(img_512,cv2.COLOR_GRAY2BGR)
    for j in contours[np.argmax(contour_length)]:
        #tmp[j[0][1]][j[0][0]] = [0,255,255]
        bool_contour[j[0][1]][j[0][0]] = True
    tmp[skeleton == 255] = [0, 0, 255]
    cv2.imshow("img_512", tmp)
    cv2.imshow("pseudo_reg", cv2.applyColorMap(img_512, cv2.COLORMAP_JET))

    #traversal_visualize(skeleton, bool_contour)
    traversal(img_512,skeleton,bool_contour,patient_ids[i])
    #cv2.waitKey(0)



