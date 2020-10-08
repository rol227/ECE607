import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi 

def KDE(data, h, training_set):
    temp_sum = [0,0,0,0,0,0,0,0,0,0]
    for i in range(0, 10):
        for item in training_set:
            if item[len(item)-1] == (i + 1):
                temp_sum[i] += guass_kernel(data, h, item, i)
        temp_sum[i] = temp_sum[i] / 300
    return temp_sum.index(max(temp_sum)) + 1; 

def guass_kernel(data, h, item, i):
    data = np.array(data)
    item = np.array(item)
    prob = ( 1 / (2*pi*(h**2))**((len(data) - 1)/2)) * math.exp(-1 * (np.linalg.norm(data[0:(len(data)-1)] - item[0:(len(data)-1)])**2) / (2*(h**2)))
    return prob;