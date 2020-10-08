import numpy as np
import matplotlib.pyplot as plt 
import csv
import Naive_Bayes_Zip_Defs as d
import timeit

start = timeit.default_timer() 


cov_mat = []
mean_mat = []
prob_mat = []
successes = []

with open('zipcode_train.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

with open('zipcode_test.csv', newline='') as csvfile:
    test_data = list(csv.reader(csvfile))

#formats the incoming data
for i in range(len(data)):
    for j in range(len(data[0])):
        data[i][j] = float(data[i][j])
        test_data[i][j] = float(test_data[i][j])

# test = np.array(d.covMatrix(data))
# # inv = np.linalg.inv(test)
# print(test)

#array of cov matrices and mean vectors
for i in range(1, 11):
    temp_list = []
    for item in data:
        if item[len(data[0]) - 1] == i:
            temp_list.append(item)
    cov_mat.append(d.covMatrix(temp_list))
    mean_mat.append(d.mean(temp_list))

cov_mat = np.array(cov_mat)
mean_mat = np.array(mean_mat)

for item in test_data:
    prob_mat.append(d.guassian(item, mean_mat, cov_mat))

for i in range(len(prob_mat)):
    if prob_mat[i] == test_data[i][len(data[i]) - 1]:
        successes.append(1)
    else:
        successes.append(0)

print(sum(successes) / len(successes))

stop = timeit.default_timer()
print('Time:', stop - start)
