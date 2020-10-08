import numpy as np
from statistics import mode

def kNN(item, k, data):
    dist_array = np.array([])
    class_list = []
    k_elements = []
    z = item[0:len(item)-1]
    for i in range(len(data)):
        x = data[i][0:(len(data[i])-1)]
        dist_array = np.append(dist_array, [np.linalg.norm(np.subtract(x,z))])
    k_elements = np.sort(dist_array)
    k_elements = k_elements[:k]
    dist_array = np.ndarray.tolist(dist_array)
    for i in range(len(k_elements)):
        class_list.append(dist_array.index(k_elements[i]))
        class_list[i] = data[class_list[i]][len(item)-1]
    return mode(class_list);