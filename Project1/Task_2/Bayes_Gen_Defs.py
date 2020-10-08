import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi 

sumxx, sumxy, sumyy = 0, 0, 0

def covariance2D (mean, data):
    sumxx, sumxy, sumyy = 0, 0, 0
    x = [[0,0],[0,0]]
    for i in range(len(data)):
        sumxx = ((data[i][0] - mean[0]) * (data[i][0] - mean[0])) + sumxx
        sumxy = ((data[i][0] - mean[0]) * (data[i][1] - mean[1])) + sumxy
        sumyy = ((data[i][1] - mean[1]) * (data[i][1] - mean[1])) + sumyy
    x[0][0] = sumxx / len(data)
    x[0][1] = sumxy / len(data)
    x[1][0] = sumxy / len(data)
    x[1][1] = sumyy / len(data)
    return x;

def mean_data (data, column1, column2):
    total1, total2 = 0, 0
    mean = [0, 0]
    for i in range(len(data)):
        total1 = data[i][column1] + total1
        total2 = data[i][column2] + total2
    mean[0] = total1 / len(data)
    mean[1] = total2 / len(data)
    return mean;

def gaussian_pX (x, cov, mean):
    A = np.matmul((x - mean), np.linalg.inv(cov))
    B = np.matmul(A, (x-mean).transpose())
    prob = ((( 1 / ((2 * pi) * (np.linalg.det(cov) ** 0.5))) * math.exp(-0.5 * (B))))
    return prob;