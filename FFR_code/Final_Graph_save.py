import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pickle



# graph_img=cv2.imread('2nd data\\1892844\\FFR_pullback.jpg')
# pullback 없는거 1891170 739092
patend_ids=os.listdir('../2nd data')
patend_ids.remove('1891170')
patend_ids.remove('739092')
graph_path = [os.path.join('..','2nd data',i,'FFR_pullback.jpg') for i in patend_ids]

def make_graph(graph_img):
    #cv2.imshow('graph_img', graph_img)
    graph_img=graph_img[170:,:850,:]
    #print(graph_img.shape)
    cv2.imshow('graph_crop_img', graph_img)

    graph_hsv=cv2.cvtColor(graph_img,cv2.COLOR_BGR2HSV)

    yellow_lower = np.array([22, 93, 200])
    yellow_upper = np.array([45, 255, 255])

    blue_lower = np.array([100, 0, 0])
    blue_upper = np.array([140, 255, 255])

    yellow_mask=cv2.inRange(graph_hsv,yellow_lower,yellow_upper)

    blue_mask=cv2.inRange(graph_hsv,blue_lower,blue_upper)
    #extract_graph=cv2.cvtColor(blue_mask,cv2.COLOR_GRAY2BGR)

    yellow_mask=cv2.inRange(graph_hsv,yellow_lower,yellow_upper)
    yellow_mask[-20:,:] = 0 # remove noise

    # cv2.imshow("blue_mask",cv2.cvtColor(blue_mask,cv2.COLOR_GRAY2BGR))
    # cv2.imshow("yellow_mask", cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR))

    for i in range(yellow_mask.shape[1]):
        if yellow_mask[:,i].sum()>10000:
            # for j in range(3):
            #     extract_graph[:,i,j]=0
            yellow_mask[:,i]=0

    L=[]
    for i in range(blue_mask.shape[0]):
        if blue_mask[i,:].sum()>100000:
            # extract_graph[i,:,0] = 0;
            # extract_graph[i,:,1] = 0;
            # extract_graph[i,:,2] = 255;
            L.append(i)    
        #print(i,extract_graph[i,:,0].sum())

    res=[]
    res.append(L[0])
    for i in range(len(L)-1):
        if L[i+1]-L[i]<3:
            pass
        else:
            res.append(L[i+1])

    #print(L)
    print(res)
    print(len(res))

    lowwer_bound=res[-1]-2
    upper_bound=res[0]+2

    length=lowwer_bound-upper_bound
    print(lowwer_bound,upper_bound,length)

    bool_mask=yellow_mask>0
    x=[]
    y=[]
    for j in range(bool_mask.shape[1]):
        for i in range(bool_mask.shape[0]):
            if bool_mask[i][j]:
                x.append(j)
                y.append(i)
                break

    points=np.array(list(zip(x,y)),np.int32)
    print((yellow_mask>0).shape)

    interpolation_img=np.zeros((bool_mask.shape[0],bool_mask.shape[1]),dtype=np.uint8)
    interpolation_img=cv2.polylines(interpolation_img,[points],False,255,1)

    x=[]
    y=[]
    for i in range(interpolation_img.shape[1]):
        #if yellow_mask[:,i].sum()>0:
        #print(i,yellow_mask[:,i].sum())
        for j in range(interpolation_img.shape[0]):
            if interpolation_img[j,i]>0:
                x.append(i)
                y.append(1-(j-upper_bound)/length)
                break


    extract_graph=cv2.cvtColor(interpolation_img,cv2.COLOR_GRAY2BGR)
    extract_graph[lowwer_bound-1:lowwer_bound+2,:,2] = 255
    extract_graph[upper_bound-1:upper_bound+2,:,2] = 255

    #cv2.imshow('mask',yellow_mask)
    cv2.imshow('graph',extract_graph)
    # cv2.waitKey()

    ### Show as Plt ###
    # ax=plt.axes()
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    # plt.plot(x, y)
    # plt.ylim([0, 1])
    # plt.show()


    res = {"X" : x, "Y" : y}

    return res

for i,j in enumerate(graph_path):
    graph_img=cv2.imread(j)
    res = make_graph(graph_img)

    ### Save as Pickle ###
    with open(os.path.join("../generated data",patend_ids[i], 'FFR_pullback.pickle'), 'wb') as fw:
        pickle.dump(res, fw)


