import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi 

def KDE(data, h, class0, class1):
    temp_sum = [0, 0]
    for i in range(len(class0)):
        temp_sum[0] += guass_kernel(data, h, class0, i)
        temp_sum[1] += guass_kernel(data, h, class1, i)
    temp_sum[0] = temp_sum[0] / len(class0)
    temp_sum[1] = temp_sum[1] / len(class1)
    return temp_sum.index(max(temp_sum)); 

def guass_kernel(data, h, classx, i):
    data = np.array(data)
    classx = np.array(classx)
    prob = ( 1 / (2*pi*(h**2))**((len(data) - 1)/2)) * math.exp(-1 * (np.linalg.norm(data[0:(len(data)-1)] - classx[i][0:(len(data)-1)])**2) / (2*(h**2)))
    return prob;