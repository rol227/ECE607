import numpy as np
import matplotlib.pyplot as plt 
import csv
import Naive_Bayes_Gen_Defs as d
import timeit

start = timeit.default_timer() 

cov_mat = []
mean_mat = []
prob_mat = []
successes = []

with open('train.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

with open('test.csv', newline='') as csvfile:
    test_data = list(csv.reader(csvfile))

#formats the incoming data
for i in range(len(data)):
    for j in range(len(data[0])):
        data[i][j] = float(data[i][j])
        test_data[i][j] = float(test_data[i][j])

#array of cov matrices and mean vectors
for i in range(0, 2):
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

test_class0x = []
test_class1x = []
test_class0y = []
test_class1y = []

for row in range(len(test_data)):
    if prob_mat[row] == 1:
        test_class1x.append(test_data[row][0])
        test_class1y.append(test_data[row][1])
    else:
        test_class0x.append(test_data[row][0])
        test_class0y.append(test_data[row][1])

plt.scatter(test_class0x, test_class0y, c='red')
plt.scatter(test_class1x, test_class1y, c='blue')
plt.show()

