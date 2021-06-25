import cv2
import numpy as np

graph_img=cv2.imread('2nd data\\29563\\FFR_pullback.jpg')
print(graph_img.shape)
graph_img=graph_img[:,:850,:]
# 843x586
graph_hsv=cv2.cvtColor(graph_img,cv2.COLOR_BGR2HSV)

yellow_lower = np.array([22, 93, 200])
yellow_upper = np.array([45, 255, 255])

yellow_mask=cv2.inRange(graph_hsv,yellow_lower,yellow_upper)
# extract_graph=cv2.cvtColor(yellow_mask,cv2.COLOR_GRAY2BGR)

# for i in range(yellow_mask.shape[1]):
#     if yellow_mask[:,i].sum()>10000:
#         for j in range(3):
#             extract_graph[:,i,j]=0
#     print(i,yellow_mask[:,i].sum())

for i in range(yellow_mask.shape[1]):
    if yellow_mask[:,i].sum()>10000:
        # for j in range(3):
        #     extract_graph[:,i,j]=0
        yellow_mask[:,i]=0
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

cv2.imshow('img',graph_img)
cv2.imshow('mask',yellow_mask)
cv2.imshow('interpolation',interpolation_img)

cv2.waitKey()