import matplotlib.pyplot as plt 
import csv
import numpy as np
import timeit

with open('train.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

with open('test.csv', newline='') as csvfile:
    test_data = list(csv.reader(csvfile))

for i in range(len(data)):
    for j in range(len(data[0])):
        data[i][j] = float(data[i][j])
        test_data[i][j] = float(test_data[i][j])

test_class0x = []
test_class1x = []
test_class0y = []
test_class1y = []

for row in range(len(data)):
    if data[row][2] == 1:
        test_class1x.append(data[row][0])
        test_class1y.append(data[row][1])
    else:
        test_class0x.append(data[row][0])
        test_class0y.append(data[row][1])

plt.scatter(test_class0x, test_class0y, c='red')
plt.scatter(test_class1x, test_class1y, c='blue')
plt.show()








