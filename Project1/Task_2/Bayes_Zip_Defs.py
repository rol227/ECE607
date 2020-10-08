import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi 

def covMatrix(data):
    C = np.zeros((len(data[0]) - 1, len(data[0]) - 1)) #initialize matrix of n x n
    for k in range(len(data[0]) - 1): #loop by number of columns once to get Cov Rows
        E_a, total_a = 0, 0        
        for l in range(len(data)):
            total_a += data[l][k]
        E_a = total_a / len(data)
        for i in range(len(data[0]) - 1): #loop by number of columns twice for n by n matrix with Cov Columns
            E_ab, E_b, total_b, total_ab = 0, 0, 0, 0
            for j in range(len(data)): #loop by number of rows
                total_b += data[j][i]
                total_ab += data[j][k] * data[j][i]
            E_ab = total_ab / len(data)
            E_b = total_b / len(data)
            C[k][i] = (E_ab) - (E_a * E_b)
            C = np.array(C)
    return C;

def mean(data):
    mean = []
    for k in range (len(data[0]) - 1):
        total_col = 0
        for l in range(len(data)):
            total_col += data[l][k]
        mean.append(total_col / len(data))
    return mean;

def zeroDetMat(mat):
    for i in range(0,(np.shape(mat)[0])):
        if mat[i,i] == 0:
            mat[i,i] += 0.0001
    return mat;

def guassian(data, mean, cov):
    temp_prob = []
    for i in range(0, 10):
        if np.linalg.det(cov[i]) == 0:
            cov[i] = zeroDetMat(cov[i])
        A = data[0:(len(data) - 1)] - mean[i]
        B = np.linalg.inv(cov[i])
        C = np.matmul(A, B)
        C = np.matmul(C, np.transpose(A))
        denom = ((2 * pi) ** (len(data) - 1)) * (np.linalg.det(cov[i]))
        temp_prob.append(( 1 / denom) * math.exp(-0.5 * C))
    return (temp_prob.index(max(temp_prob)) + 1)
        