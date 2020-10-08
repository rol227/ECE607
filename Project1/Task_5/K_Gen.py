import matplotlib.pyplot as plt 
import csv
import numpy as np
import K_Gen_Defs as d
import timeit

start = timeit.default_timer()

class_est = []
successes = []
k = 5

with open('train.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

with open('test.csv', newline='') as csvfile:
    test_data = list(csv.reader(csvfile))

#formats the incoming data as floats
for i in range(len(data)):
    for j in range(len(data[0])):
        data[i][j] = float(data[i][j])
        test_data[i][j] = float(test_data[i][j])

for item in test_data: 
    class_est.append(d.kNN(item, k, data))

for i in range(len(class_est)):
    if class_est[i] == test_data[i][len(test_data[i]) - 1]:
        successes.append(1)
    else:
        successes.append(0)

print(sum(successes) / len(successes))

stop = timeit.default_timer()
print('Time:', stop - start)

test_class0x = []
test_class1x = []
test_class0y = []
test_class1y = []

for row in range(len(test_data)):
    if class_est[row] == 1:
        test_class1x.append(test_data[row][0])
        test_class1y.append(test_data[row][1])
    else:
        test_class0x.append(test_data[row][0])
        test_class0y.append(test_data[row][1])

plt.scatter(test_class0x, test_class0y, c='red')
plt.scatter(test_class1x, test_class1y, c='blue')
plt.show()