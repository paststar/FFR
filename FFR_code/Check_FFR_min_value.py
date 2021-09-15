import os
import pickle

patient_ids = os.listdir('../2nd data')
patient_ids.remove('1891170')  # no pull back
patient_ids.remove('739092')  # no pull back

min_FFR_value = {
    "1538839" : 0.9,
    "834572" : 0.94,
    "232209" : 0.95,
    "898265" : 0.89,
    "1129840" : 0.76,
    "190371" : 0.81,
    "744682" : 0.80,
    "1645701" : 0.69,
    "1745659" : 0.75, # read graph
    "1908822" : 0.9,
    "347576" : 0.98,
    "785547" : 0.92,
    "1892844" : 0.85,
    "699639" : 0.77,
    "29563" : 0.75,
    "1765238" : 0.82,
    "1844665" : 0.87,
    "1470403" : 0.89,
    "415865" : 0.69
}


for i in range(len(patient_ids)):

    with open(os.path.join("../generated data",patient_ids[i], 'FFR_pullback.pickle'), 'rb') as fr:
         D = pickle.load(fr)
         true_x = D['X']
         true_y = D['Y']

    if abs(min_FFR_value[patient_ids[i]]-min(true_y)>0.01):
        print(patient_ids[i],min_FFR_value[patient_ids[i]], min(true_y))

#print(len(patient_ids),len(min_FFR_value))



