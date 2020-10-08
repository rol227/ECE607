import matplotlib.pyplot as plt 
import csv
import numpy as np
import K_Zip_Defs as d
import timeit

start = timeit.default_timer()

class_est = []
successes = []
k = 5

with open('zipcode_train.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

with open('zipcode_test.csv', newline='') as csvfile:
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

print('Success Rate:', sum(successes) / len(successes))

stop = timeit.default_timer()
print('Time:', stop - start)