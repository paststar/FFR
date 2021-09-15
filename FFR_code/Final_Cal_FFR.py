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
#patient_ids.remove('1908822') # bad translation
# patient_ids.remove('785547')
# patient_ids.remove('415865') # no reg.png (registration)

images_path = [(os.path.join('..', '2nd data', i, 'AP', 'AP_QCA.bmp'),
                os.path.join('..', '2nd data', i, 'AP', 'AP_contrast.bmp'),
                os.path.join('..', '2nd data', i, 'AP', 'AP_pre.bmp')) for i in patient_ids]
before_reg_path = [os.path.join('..', '2nd data', i, 'AP', 'AP_reg.png') for i in patient_ids]
#reg_images_path = [os.path.join('..',"generated data",i,"nongrid2_reg.png") for i in patient_ids]
reg_images_path = [os.path.join('..',"generated data",i,"bottom_hat.png") for i in patient_ids]

# reg_images_path = [os.path.join('..',"generated data",i,"multimodal_translation_reg.png") for i in patient_ids]
# reg_images_path = [os.path.join('..', "generated data", i, "nongrid2_activecontour_superpixel_mean.png") for i in patient_ids]
skeleton_path=[os.path.join('..',"generated data",i,"active_contour_skeleton.png") for i in patient_ids]

graph_path = [os.path.join('..', '2nd data', i, 'FFR_pullback.jpg') for i in patient_ids]

vessel_mask_path = [(os.path.join('..', "generated data", i, "superpixels_intersection.png"),
                     os.path.join('..', "generated data", i, "active_contour_mask.png"),
                     os.path.join('..', "generated data", i, "QCA_mask.png")) for i in patient_ids]
# vessel_mask_path=[os.path.join('..',"generated data",i,"active_contour_mask.png") for i in patient_ids]
# vessel_mask_path=[os.path.join('..',"generated data",i,"QCA_mask.png") for i in patient_ids]

res_pickle = [
    # 'nongrid2_reg_active_contour_mask_centerline_density.pickle',
    # 'nongrid2_reg_QCA_mask_centerline_density.pickle',
    # 'nongrid2_activecontour_superpixel_mean_centerline.pickle',
    # 'multimodal_translation_active_contour_mask_centerline_density.pickle',
    #  'multimodal_translation_QCA_mask_centerline_density.pickle',
    #  'multimodal_translation_activecontour_superpixel_mean_centerline.pickle',
    # 'bottom_hat_active_contour_mask_centerline_density.pickle',
    # 'bottom_hat_QCA_mask_centerline_density.pickle',
    'bottom_hat_activecontour_superpixel_mean_centerline.pickle'
]

def curve_fit_ffr(num, X, Y, true_x, true_y):
    ### remove head and back ###
    x = X[10:-10]
    y = Y[10:-10]

    #print(len(X),len(x))

    end = np.argmax(y)

    # print(x[end],y[end])
    # print(x)
    # print(y)

    Y = Y / y[end]
    x = x[end:]
    y = y[end:] / y[end]

    ### curve fitting ###
    def func(x, a, b, c, d, e, f, g):
        return a * (x ** 6) + b * (x ** 5) + c * (x ** 4) + d * (x ** 3) + e * (x ** 2) + f * x + g

    def first_derivative_func(x, a, b, c, d, e, f, g):
        return 6 * a * (x ** 5) + 5 * b * (x ** 4) + 4 * c * (x ** 3) + 3 * d * (x**2) + 2 * e * x + f

    def second_derivative_func(x, a, b, c, d, e, f, g):
        return 30 * a * (x ** 4) + 20 * b * (x ** 3) + 12 * c * (x ** 2) + 6 * d * x + 2 * e


    # def func(x, a, b, c, d, e):
    #     return a * (x ** 4) + b * (x ** 3) + c * (x ** 2) + d * x  + e
    #
    # def first_derivative_func(x, a, b, c, d, e):
    #     return 4 * a * (x ** 3) + 3 * b * (x ** 2) + 2 * c * x + d
    #
    # def second_derivative_func(x, a, b, c, d, e):
    #     return 12 * a * (x ** 2) + 6 * b * x + 2 * c

    popt, _ = curve_fit(func, x, y)

    ###### predict ######
    saddle_point = np.roots([(len(popt) - i - 1) * popt[i] for i in range(len(popt) - 1)])
    saddle_point = saddle_point[(lambda x: x.imag == 0.0)(saddle_point)].real
    saddle_point = saddle_point[(saddle_point>x[-1]) & (saddle_point<x[0])] # inrange

    predict_val = []

    ### use local minmum/maximum ###
    flag1 = False
    for i in saddle_point[::-1]:
        if second_derivative_func(i, *popt) > 0:
            right_local_minimum = func(i, *popt)
            plt.scatter(i, right_local_minimum, marker='s', s=100, color="tab:green")
            predict_val.append(right_local_minimum / max(y))  #### method 1
            predict_val.append(right_local_minimum)           #### method 2
            flag1 = True
            #print("predict : ", predict_val)
            break

    if not flag1 or len(saddle_point)==0:
        print("eroor : "+patient_ids[num])
        #patient_ids.remove(patient_ids[num])
        return 1
    # assert flag1

    ### local maximum은 없는 경우 있음 ###
    # flag2 = False
    # for i in saddle_point[::-1]:
    #     if second_derivative_func(i, *popt) < 0:
    #         left_local_maximum = func(i, *popt)
    #         plt.scatter(i, left_local_maximum, marker="s", color = "black")
    #
    #         predict_val.append(left_local_maximum / max(y))
    #         predict_val.append(left_local_maximum)
    #         flag2 = True
    #         #print("predict : ", predict_val)
    #         break
    #
    # if flag1 and flag2:
    #     predict_val.append(left_local_maximum/right_local_minimum)   #### method 5

    ### use first derivate ###
    tmp_x = np.linspace(int(saddle_point[-1]), int(saddle_point[0]),
                        num=(int(saddle_point[0]) - int(saddle_point[-1]) + 1) * 10 + 1)
    first_derivative_val = np.abs(first_derivative_func(tmp_x, *popt))
    first_x = np.array([tmp_x[np.argmax(first_derivative_val)], tmp_x[np.argmin(first_derivative_val)]])
    first_y = func(first_x, *popt)

    plt.scatter(first_x[0], first_y[0], marker='s', s=100, color="tab:red")
    # #plt.scatter(first_x[1], first_y[1], marker='s', s=100, color="tab:green")

    predict_val.append(first_y[0])  #### method 3
    predict_val.append(first_y[1])  #### method 4

    ### use second derivate ###
    tmp_x = np.linspace(int(saddle_point[-1]), int(saddle_point[0]), num=(int(saddle_point[0]) - int(saddle_point[-1]) + 1) * 10 + 1)
    second_derivative_val = second_derivative_func(tmp_x, *popt)
    second_x = np.array([tmp_x[np.argmax(second_derivative_val)],tmp_x[np.argmin(second_derivative_val)]])
    second_y = func(second_x,*popt)

    #plt.scatter(second_x[0], second_y[0], marker='s', s=100, color="tab:red")
    #plt.scatter(second_x[1],second_y[1],marker='s', color = "blue")

    predict_val.append(second_y[0])         #### method 5
    predict_val.append(second_y[1])         #### method 6

    #predict_val.append(second_y[0] / func(saddle_point[0], *popt))  #### method
    #predict_val.append(second_y[0]/right_local_minimum)

    ### real FFR ###
    real.append(np.min(true_y))

    ### predicted FFR ###
    tmp_x = np.linspace(min(x), max(x), num=(max(x) - min(x) + 1) * 10 + 1)
    predict_val.append(first_y[0]/max(func(tmp_x, *popt))) # 7
    predict_val.append(second_y[0]/max(func(tmp_x, *popt))) # 8
    predict_val.append(right_local_minimum / max(func(tmp_x, *popt))) # 9
    predict.append(predict_val)

    ### visualize ###
    # plt.title(f"patient id: {id} \n predict : {[ '{0:.4f}'.format(val) for val in predict_val]}")
    # plt.xlim(0, max(X))
    # #plt.ylim(0, 1)
    #
    # plt.scatter(X, Y, marker='.', s= 30, color="tab:cyan") # origin
    # plt.scatter(x, y, marker='.', s= 30, color='tab:blue') # Remove front/back
    #
    # tmp_x = np.linspace(min(x), max(x), num=(max(x) - min(x) + 1) * 10 + 1)
    # plt.plot(tmp_x, func(tmp_x, *popt), color='tab:gray', linewidth=3) # Curve visualize
    # plt.show()

    ### RANSAC ###
    # reg = RANSACRegressor(random_state=0).fit(np.expand_dims(X,axis=1), Y)
    # x = np.linspace(min(X), max(X), num=(max(X)-min(X)+1) * 10 + 1)
    # y = reg.predict(np.expand_dims(x,axis=1))
    # print(y)
    # plt.plot(x,y)
    # plt.show()

def marking(num, target):
    skeleton = cv2.imread(skeleton_path[num],0)
    bool_skel = skeleton == 255
    visited = bool_skel.copy()
    cordinate_skel = []

    d_x = [-1, 0, 1, 0, -1, 1, 1, -1]
    d_y = [0, 1, 0, -1, 1, 1, -1, -1]

    def check_range(tmp):
        if(0<=tmp and tmp<512):
            return True
        else:
            return False

    for i in range(visited.shape[0]):
        for j in range(visited.shape[1]):
            if(visited[i][j]):
                #print("start : ",i,j)
                visited[i][j] = False
                queue = deque([(i,j)])

                while queue:
                    x,y = queue.popleft()
                    cordinate_skel.append((y,x))

                    for k in range(8):
                        tmp_x = x + d_x[k]
                        tmp_y = y + d_y[k]
                        if(not (check_range(tmp_x) and check_range(tmp_y))):
                            continue

                        if(visited[tmp_x][tmp_y]):
                            queue.append((tmp_x,tmp_y))
                            visited[tmp_x][tmp_y] = False

    X = [0]
    for i in range(1,len(cordinate_skel)):
        X.append(X[i-1] + abs(cordinate_skel[i][0]-cordinate_skel[i-1][0]) + abs(cordinate_skel[i][1]-cordinate_skel[i-1][1]))

    X = X[-1] - np.array(X)
    X = X[::-1]
    cordinate_skel = cordinate_skel[::-1]

    def binary_search(tar):
        left = 0
        right = len(X)-1
        res = []
        while left <= right:
            center = (left + right) // 2

            if X[center] == tar:
                res.append(center)
                return res
            elif X[center] > tar:
                right = center - 1
            else:
                left = center + 1

        res.append(left)
        res.append(right)
        return res

    color_PaPd = {"Pd" : [0,0,255],"Pa":[0,255,0]}
    reg_img = cv2.imread(reg_images_path[num], 0)
    marking_img = cv2.cvtColor(reg_img,cv2.COLOR_GRAY2BGR)
    marking_pseudo_img = cv2.applyColorMap(reg_img, cv2.COLORMAP_JET)

    marking_img[skeleton==255] = [0,255,255]
    #cv2.imshow("pseudo image", cv2.applyColorMap(cv2.imread(reg_images_path[i], 0), cv2.COLORMAP_JET))

    for i in ["Pa","Pd"]:
        if i in target:
            res_ind = binary_search(target[i])


            if len(res_ind) == 1:
                center = cordinate_skel[res_ind[0]]

            elif len(res_ind) == 2:
                if abs(X[res_ind[0]] - X[res_ind[1]])>2:
                    pass
                    #raise ValueError

                center = cordinate_skel[res_ind[0]]

            cv2.circle(marking_img, center, 10, color_PaPd[i], 2)
            cv2.circle(marking_pseudo_img, center, 10, color_PaPd[i], 2)

    #cv2.imshow("marking_img",marking_img)
    #cv2.imshow("marking_pseudo_img", marking_pseudo_img)

    fig = plt.figure(figsize=(10, 5))
    plt.suptitle(f"patient id: {patient_ids[num]}", fontsize=15)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.imshow(cv2.cvtColor(marking_img, cv2.COLOR_BGR2RGB),interpolation='nearest')
    ax1.set_title('marking img')
    ax1.axis("off")

    ax2.imshow(cv2.cvtColor(marking_pseudo_img, cv2.COLOR_BGR2RGB),interpolation='nearest')
    ax2.set_title('marking pseudo img')
    ax2.axis("off")
    plt.savefig(os.path.join("/home/bang/Desktop/plt_save/bottom-hat-superpixel-mean + median",patient_ids[num]+"_marking.png"))

def cal_ffr(num, X, Y, true_x, true_y, mode,degree = 4, median = 0):
    ### real FFR ###
    real.append(np.min(true_y))


    fig = plt.figure(figsize=(10,5))
    plt.suptitle(f"patient id: {patient_ids[num]}", fontsize=15)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.set_ylim([0,1])
    ax1.plot(true_x, true_y, color="black", linestyle="-",linewidth=3)
    ax1.set_title(f"GT \n Real FFR : {min(true_y)}")
    # f"predict : {[ '{0:.4f}'.format(val) for val in predict_val]}")

    ### remove head and back ###
    #x = X[10:-10]
    #y = Y[10:-10]
    x = X[5:-5]
    y = Y[5:-5]
    #x=X
    #y=Y
    # print(len(X),len(x))
    # print(x[end],y[end])
    # print(x)
    # print(y)

    end = np.argmax(y)
    x = x[end:]
    y = y[end:]

    ### normalize ###
    y = (y-min(y))/(max(y)-min(y))

    ### median filtering ###
    def median_filter(xx,yy,k,s=1):
        t=k//2
        c=t
        res_x=[]
        res_y=[]

        while c+t<=len(xx)-1:
            res_x.append(xx[c])
            res_y.append(np.median(yy[c-t:c+t+1]))
            #print(yy[c-t:c+t+1])
            c+=s

        return  np.array(res_x),np.array(res_y)

    if median!=0:
        x, y = median_filter(x, y, median)

    #estimators = [
        #('OLS', LinearRegression()),
        #('Theil-Sen', TheilSenRegressor(random_state=42)),
        #('RANSAC', RANSACRegressor(random_state=42))
        #('HuberRegressor', HuberRegressor())
    #]

    if mode == "OLS":
        estimators = [
            ('OLS', LinearRegression())
        ]

    elif mode == "RANSAC":
        estimators = [
            ('RANSAC', RANSACRegressor(random_state=42))
        ]

    colors = {'OLS': 'lightgreen', 'Theil-Sen': 'gold', 'RANSAC': 'black', 'HuberRegressor': 'turquoise'}
    linestyle = {'OLS': '--', 'Theil-Sen': '-.', 'RANSAC': '-', 'HuberRegressor': '--'}
    lw = 3

    x_plot = np.linspace(x.min(), x.max())

    odel_param = []
    for name, estimator in estimators:
        model = make_pipeline(PolynomialFeatures(degree), estimator)
        model.fit(x[:,np.newaxis], y)
        y_plot = model.predict(x_plot[:, np.newaxis])
        #plt.plot(x_plot, y_plot, color=colors[name], linestyle=linestyle[name],linewidth=lw, label='%s' % (name))

        if name == "RANSAC":
            #print(model.steps[1][1].estimator_.coef_)
            #plt.plot(x_plot, y_plot, color=colors[name], linestyle=linestyle[name], linewidth=lw, label='%s' % (name))
            popt = model.steps[1][1].estimator_.coef_[::-1]
            popt[-1] = model.steps[1][1].estimator_.intercept_
            #print(dir(model.steps[1][1].estimator_))

        if name == "OLS":
            #print(model.steps[1][1].coef_[::-1])
            popt = model.steps[1][1].coef_[::-1]
            popt[-1] = model.steps[1][1].intercept_

    ### visualize regression ###
    # plt.scatter(X, Y, marker='.', s= 30, color="tab:cyan") # origin
    # plt.scatter(x, y, marker='.', s= 30, color='tab:orange') # Remove front/back
    # legend = plt.legend(loc='upper right', frameon=False,
    #                    prop=dict(size='x-small'))
    # plt.show()
    #print(popt)

    ### curve fitting ###
    if degree == 6:
        def func(x, a, b, c, d, e, f, g):
            return a * (x ** 6) + b * (x ** 5) + c * (x ** 4) + d * (x ** 3) + e * (x ** 2) + f * x + g

        def first_derivative_func(x, a, b, c, d, e, f, g):
            return 6 * a * (x ** 5) + 5 * b * (x ** 4) + 4 * c * (x ** 3) + 3 * d * (x ** 2) + 2 * e * x + f

        def second_derivative_func(x, a, b, c, d, e, f, g):
            return 30 * a * (x ** 4) + 20 * b * (x ** 3) + 12 * c * (x ** 2) + 6 * d * x + 2 * e

    elif degree == 4:
        def func(x, a, b, c, d, e):
            return a * (x ** 4) + b * (x ** 3) + c * (x ** 2) + d * x  + e

        def first_derivative_func(x, a, b, c, d, e):
            return 4 * a * (x ** 3) + 3 * b * (x ** 2) + 2 * c * x + d

        def second_derivative_func(x, a, b, c, d, e):
            return 12 * a * (x ** 2) + 6 * b * x + 2 * c

    ###### predict ######
    saddle_point = np.roots([(len(popt) - i - 1) * popt[i] for i in range(len(popt) - 1)])
    saddle_point = saddle_point[(lambda x: x.imag == 0.0)(saddle_point)].real
    saddle_point = saddle_point[(saddle_point > x[-1]) & (saddle_point < x[0])]  # inrange

    predict_val = []

    ### use local minmum/maximum ###
    # flag1 = False
    # for i in saddle_point[::-1]:
    #     if second_derivative_func(i, *popt) > 0:
    #         right_local_minimum = func(i, *popt)
    #         # plt.scatter(i, right_local_minimum, marker='s', s=100, color="tab:green")
    #         # predict_val.append(right_local_minimum / max(y))  #### method 1
    #         # predict_val.append(right_local_minimum)  #### method 2
    #         flag1 = True
    #         # print("predict : ", predict_val)
    #         break
    #
    # if not flag1:
    #     print("no right local minimum eroor : " + patient_ids[num])
    #     #return 1

    if  len(saddle_point) == 0:
        print("no saddle point eroor : " + patient_ids[num])
        tmp_x = np.linspace(min(x), max(x), num=(max(x) - min(x) + 1) * 10 + 1)
        ax2.plot(tmp_x, func(tmp_x, *popt), color='tab:gray', linewidth=3)  # Curve visualize
        #return 1

    elif len(saddle_point) == 1:
        print("one saddle point : " + patient_ids[num])
        tmp_x = np.linspace(min(x), max(x), num=(max(x) - min(x) + 1) * 10 + 1)
        ax2.plot(tmp_x, func(tmp_x, *popt), color='tab:gray', linewidth=3)  # Curve visualize

        # tmp_x = np.linspace(int(saddle_point[-1]), int(saddle_point[0]),
        #                     num=(int(saddle_point[0]) - int(saddle_point[-1]) + 1) * 10 + 1)
        # plt.plot(tmp_x, func(tmp_x, *popt), color='lightgreen', linewidth=3)  # Curve visualize
        # graph_x = np.linspace(min(x), int(saddle_point[-1]))
        # plt.plot(graph_x, func(graph_x, *popt), color='lightgreen', linewidth=3, linestyle="--")  # Curve visualize
        # graph_x = np.linspace(int(saddle_point[0]), max(x))
        # plt.plot(graph_x, func(graph_x, *popt), color='lightgreen', linewidth=3, linestyle="--")  # Curve visualize

    else:
        tmp_x = np.linspace(min(x), max(x), num=(max(x) - min(x) + 1) * 10 + 1)
        ax2.plot(tmp_x, func(tmp_x, *popt), color='tab:gray', linewidth=3)  # Curve visualize

        # tmp_x = np.linspace(int(saddle_point[-1]), int(saddle_point[0]),
        #                     num=(int(saddle_point[0]) - int(saddle_point[-1]) + 1) * 10 + 1)
        # plt.plot(tmp_x, func(tmp_x, *popt), color='tab:gray', linewidth=3)  # Curve visualize
        # graph_x = np.linspace(min(x), int(saddle_point[-1]))
        # plt.plot(graph_x, func(graph_x, *popt), color='tab:gray', linewidth=3, linestyle="--")  # Curve visualize
        # graph_x = np.linspace(int(saddle_point[0]), max(x))
        # plt.plot(graph_x, func(graph_x, *popt), color='tab:gray', linewidth=3, linestyle="--")  # Curve visualize

    #good_patient_id.append(patient_ids[num])

    #print(saddle_point)

    ### use first derivate ###
    first_derivative_val = first_derivative_func(tmp_x, *popt)
    first_x = np.array([tmp_x[np.argmax(first_derivative_val)], tmp_x[np.argmin(first_derivative_val)]])
    first_abs_x = np.array([tmp_x[np.argmax(np.abs(first_derivative_val))], tmp_x[np.argmin(np.abs(first_derivative_val))]])

    first_y = func(first_x, *popt)
    first_abs_y = func(first_abs_x, *popt)

    predict_val.append(first_y[0])  #### method 1
    predict_val.append(first_y[1])  #### method 2
    predict_val.append(first_abs_y[0])  #### method 3
    predict_val.append(first_abs_y[1])  #### method 4

    ### use second derivate ###
    second_derivative_val = second_derivative_func(tmp_x, *popt)
    second_x = np.array([tmp_x[np.argmax(second_derivative_val)], tmp_x[np.argmin(second_derivative_val)]])
    second_abs_x = np.array([tmp_x[np.argmax(np.abs(second_derivative_val))], tmp_x[np.argmin(np.abs(second_derivative_val))]])

    second_y = func(second_x, *popt)
    second_abs_y = func(second_abs_x, *popt)

    predict_val.append(second_y[0])  #### method 5
    predict_val.append(second_y[1])  #### method 6
    predict_val.append(second_abs_y[0])  #### method 7
    predict_val.append(second_abs_y[1])  #### method 8

    ### use max###
    xx = np.linspace(min(x), max(x), num=(max(x) - min(x) + 1) * 10 + 1)

    max_func = max(func(xx, *popt))
    min_func = min(func(xx, *popt))
    bounded_max_func = min(1,max_func)

    predict_val.append(first_y[0] / max_func) #### method 9
    predict_val.append(first_y[1] / max_func) #### method 10
    predict_val.append(first_abs_y[0] / max_func) #### method 11
    predict_val.append(first_abs_y[1] / max_func)  #### method 12
    predict_val.append(second_y[0] / max_func)  #### method 13
    predict_val.append(second_y[1] / max_func)  #### method 14
    predict_val.append(second_abs_y[0] / max_func)  #### method 15
    predict_val.append(second_abs_y[1] / max_func)  #### method 16

    predict_val.append(first_y[0] / bounded_max_func)  #### method 17
    predict_val.append(first_y[1] /bounded_max_func)  #### method 18
    predict_val.append(first_abs_y[0] / bounded_max_func)  #### method 19
    predict_val.append(first_abs_y[1] / bounded_max_func)  #### method 20
    predict_val.append(second_y[0] / bounded_max_func)  #### method 21
    predict_val.append(second_y[1] / bounded_max_func)  #### method 22
    predict_val.append(second_abs_y[0] / bounded_max_func)  #### method 23
    predict_val.append(second_abs_y[1] / bounded_max_func)  #### method 24

    predict_val.append(min_func / max_func) #### method 25


    # predict_val.append(right_local_minimum / max_func)  #### method 9
    # predict_val.append(right_local_minimum / bounded_max_func)  #### method 18
    predict.append(predict_val)

    ### visualize ###
    ax2.set_title(
        #f"patient id: {id} \n "
              f"Predicted : {predict_val[15]}")
              #f"predict : {[ '{0:.4f}'.format(val) for val in predict_val]}")
    #plt.xlim(0, max(X))
    #plt.ylim(0, 1)

    #ax2.scatter(X, Y, marker='.', s= 30, color="tab:cyan") # origin
    ax2.scatter(x, y, marker='.', s= 30, color='tab:blue') # Remove front/back

    ax2.scatter(second_abs_x[1], second_abs_y[1], marker='s', s=100, color="tab:red") # FFR - method 8
    #ax2.scatter(tmp_x[np.argmin(func(xx, *popt))], min_func, marker='s', s=100, color="tab:red")  # Pd
    #ax2.scatter(tmp_x[np.argmax(func(xx, *popt))], max_func, marker='s', s=100, color="tab:green") # Pa

    target = {}
    target["Pd"] = int(second_abs_x[1])
    #target["Pa"] = int(tmp_x[np.argmax(func(xx, *popt))])
    #marking(num,target)

    #plt.scatter(second_x[0], second_y[0], marker='s', s=100, color="tab:red")
    #plt.savefig(os.path.join("/home/bang/Desktop/plt_save/bottom-hat-superpixel-mean + median",patient_ids[num]+".png"))
    plt.show()
    plt.close()

def cal_ffr_SVR(num, X, Y, true_x, true_y,  median = True):
    fig = plt.figure(figsize=(10,5))
    plt.suptitle(f"patient id: {patient_ids[num]}", fontsize=15)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.set_ylim([0,1])
    ax1.plot(true_x, true_y, color="black", linestyle="-",linewidth=3)
    ax1.set_title(f"GT \n Real FFR : {min(true_y)}")

    # f"predict : {[ '{0:.4f}'.format(val) for val in predict_val]}")

    ### remove head and back ###
    #x = X[10:-10]
    #y = Y[10:-10]
    #x = X[5:-5]
    #y = Y[5:-5]
    x=X
    y=Y

    end = np.argmax(y)

    Y = Y / y[end]
    x = x[end:]
    y = y[end:] / y[end]

    ### median filtering ###
    def median_filter(xx,yy,k,s=1):
        t=k//2
        c=t
        res_x=[]
        res_y=[]

        while c+t<=len(xx)-1:
            res_x.append(xx[c])
            res_y.append(np.median(yy[c-t:c+t+1]))
            #print(yy[c-t:c+t+1])
            c+=s

        return  res_x,res_y

    if median:
        x,y = median_filter(x,y,5)


    ### SVR ###
    svr = SVR(kernel='rbf', C=1, gamma=0.0001, epsilon=0.01)

    svr.fit(np.expand_dims(x,axis=-1),y)
    x_plot = np.linspace(min(x), max(x), num=(max(x) - min(x) + 1) * 10 + 1)
    y_plot = svr.predict(np.expand_dims(x_plot, axis=-1))
    plt.plot(x_plot, y_plot, color = "tab:gray", lw = 3)

    ### real FFR ###
    real.append(np.min(true_y))

    ### predicted FFR ###
    xx = np.linspace(min(x), max(x), num=(max(x) - min(x) + 1) * 10 + 1)

    predict_val = []
    predict_val.append(min(y_plot) / max(y_plot))
    predict.append(predict_val)


    ### visualize ###
    ax2.set_title(
        #f"patient id: {id} \n "
              f"Predicted : {predict_val}")
              #f"predict : {[ '{0:.4f}'.format(val) for val in predict_val]}")
    #plt.xlim(0, max(X))
    #plt.ylim(0, 1)

    ax2.scatter(X, Y, marker='.', s= 30, color="tab:cyan") # origin
    ax2.scatter(x, y, marker='.', s= 30, color='tab:blue') # Remove front/back

    ax2.scatter(x_plot[np.argmin(y_plot)], min(y_plot), marker='s', s=100, color="tab:red")  # Pd
    ax2.scatter(x_plot[np.argmax(y_plot)], max(y_plot), marker='s', s=100, color="tab:green")  # Pa

    target = {}
    target["Pd"] = int(x_plot[np.argmin(y_plot)])
    target["Pa"] = int(x_plot[np.argmax(y_plot)])
    #marking(num,target)

    plt.show()
    plt.close()

def cal_ffr_medain_kernel(num, X, Y, true_x, true_y, mode=None):
    fig = plt.figure(figsize=(10,5))
    plt.suptitle(f"patient id: {patient_ids[num]}", fontsize=15)
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)

    ax1.set_ylim([0,1])
    ax1.plot(true_x, true_y, color="black", linestyle="-",linewidth=3)
    ax1.set_title(f"GT \n Real FFR : {min(true_y)}")

    # f"predict : {[ '{0:.4f}'.format(val) for val in predict_val]}")

    ### remove head and back ###
    #x = X[10:-10]
    #y = Y[10:-10]
    #x = X[5:-5]
    #y = Y[5:-5]
    x=X
    y=Y

    end = np.argmax(y)

    Y = Y / y[end]
    x = x[end:]
    y = y[end:] / y[end]

    ### median filtering ###

    def median_filter(xx,yy,k,s=1):
        t=k//2
        c=t
        res_x=[]
        res_y=[]

        while c+t<=len(xx)-1:
            res_x.append(xx[c])
            res_y.append(np.median(yy[c-t:c+t+1]))
            #print(yy[c-t:c+t+1])
            c+=s

        return  res_x,res_y

    median_x,median_y = median_filter(x,y,31)
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(median_x, median_y, marker='.', s= 30, color="tab:red")
    ax3.set_ylim(0.1)

    ### real FFR ###
    real.append(np.min(true_y))

    ### predicted FFR ###
    #xx = np.linspace(min(x), max(x), num=(max(x) - min(x) + 1) * 10 + 1)

    predict_val = []

    predict.append(predict_val)


    ### visualize ###
    ax2.set_title(
        #f"patient id: {id} \n "
              f"Predicted : {predict_val}")
              #f"predict : {[ '{0:.4f}'.format(val) for val in predict_val]}")
    #plt.xlim(0, max(X))
    #plt.ylim(0, 1)

    ax2.scatter(X, Y, marker='.', s= 30, color="tab:cyan") # origin
    ax2.scatter(x, y, marker='.', s= 30, color='tab:blue') # Remove front/back

    # ax2.scatter(x_plot[np.argmin(y_plot)], min(y_plot), marker='s', s=100, color="tab:red")  # Pd
    # ax2.scatter(x_plot[np.argmax(y_plot)], max(y_plot), marker='s', s=100, color="tab:green")  # Pa

    # target = {}
    # target["Pd"] = int(x_plot[np.argmin(y_plot)])
    # target["Pa"] = int(x_plot[np.argmax(y_plot)])
    # marking(num,target)

    plt.show()
    plt.close()


if __name__ == "__main__":
    num = 0
    for j in res_pickle:
        predict = []
        real = []
        good_patient_id = []

        print("< "+j+" >")

        for i in range(len(patient_ids)):
            id = patient_ids[i]
            GT_graph = cv2.imread(graph_path[i])[155:, :1000, :]
            #cv2.imshow("pseudo image", cv2.applyColorMap(cv2.imread(reg_images_path[i],0),cv2.COLORMAP_JET))
            #cv2.imshow("graph",GT_graph)

            ### load data ###

            with open(os.path.join("../generated data", id, 'FFR_pullback.pickle'), 'rb') as fr:
                GT = pickle.load(fr)
                true_x = GT['X']
                true_y = GT['Y']

            with open(os.path.join("../generated data", id, j), 'rb') as fr:
                res = pickle.load(fr)
                X = res['X']
                Y = res['Y']

            #curve_fit_ffr(i,X,Y,true_x,true_y)
            cal_ffr(i,X,Y,true_x,true_y,"RANSAC",degree = 6,median = 2)
            #cal_ffr_SVR(i,X,Y,true_x,true_y,median = False)
            #cal_ffr_medain_kernel(i,X,Y,true_x,true_y)

        #print(real)
        #print(predict)
        real = np.array(real)
        predict = np.array(predict)

        M = 0
        M_ind = 0
        for i in range(predict.shape[-1]):
            corr, _ = pearsonr(real, predict[:,i])
            # mi = normalized_mutual_info_score(real, predict)
            #mutual_info_regression
            mi = mutual_info_regression(np.expand_dims(predict[:,i], axis=1), real,discrete_features='auto', n_neighbors=3, copy=True, random_state=None)

            if mi > 0.3 or abs(corr) > 0.6:
                print('method %d - corr : %.5f mi : %.5f' % (i + 1, corr, mi))
            #print('method %d - corr : %.5f mi : %.5f' % (i + 1, corr, mi))

            if np.abs(corr)> abs(M):
                M = corr
                M_ind = i

        print()
        print("### Best Correlation ###")
        print(f"method: {M_ind + 1} Corr : {M:.4f}")

        fig, ax = plt.subplots()
        ax.scatter(predict[:,M_ind], real, marker="s", color="tab:blue")
        plt.xlabel("Predict")
        plt.ylabel("Real")
        #plt.xlim(0, 1)
        #plt.ylim(0, 1)
        plt.title(f"method: {M_ind+1} \n Corr : {M:.4f}")
        regr = LinearRegression()

        regr.fit(np.expand_dims(predict[:, M_ind], axis=1), real)
        plt.plot(predict[:, M_ind], regr.predict(np.expand_dims(predict[:, M_ind], axis=1)),
                 linewidth=2, color="tab:red")

        for i, txt in enumerate(patient_ids):
            ax.annotate(txt, (predict[i][M_ind], real[i]))
            #ax.annotate(str(i+1), (predict[i][M_ind], real[i]))

        print()
        plt.show()






