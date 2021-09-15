import os
import cv2
import numpy as np
import pickle
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

patient_ids=os.listdir('../2nd data')
patient_ids.remove('1891170') # no pull back graph
patient_ids.remove('739092') # no pull back graph

graph_path = [os.path.join('..','2nd data',i,'FFR_pullback.jpg') for i in patient_ids]


def func(x, a, b, c, d, e, f, g, h ,i):
    return a * (x ** 8) + b * (x ** 7) + c * (x ** 6) + d * (x ** 5) + e * (x ** 4) + f * (x ** 3) + g * (x ** 2) +h * x + i

# def func(x, a, b, c, d, e, f, g):
#     return a * (x ** 6) + b * (x ** 5) + c * (x ** 4) + d * (x ** 3) + e * (x ** 2) + f * x + g

for i in range(len(patient_ids)):
    print(f"id : {patient_ids[i]}")

    graph_img = cv2.imread(graph_path[i])
    cv2.imshow("graph_img", graph_img)

    with open(os.path.join("../generated data",patient_ids[i], 'FFR_pullback.pickle'), 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    Y = data['Y']

    popt, _ = curve_fit(func, X, Y)
    x = np.linspace(0, max(X), num=max(X) * 10 + 1)
    #plt.ylim(0, 1)
    plt.scatter(X, Y, marker='.')
    plt.plot(x, func(x, *popt), color='red', linewidth=2)
    plt.show(block=False)

    cv2.waitKey(0)