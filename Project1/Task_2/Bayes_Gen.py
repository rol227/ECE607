import numpy as np
import matplotlib.pyplot as plt 
import csv
import Bayes_Gen_Defs as d
import timeit

start = timeit.default_timer() 

full_set = []
class0 = []
class1 = []

test_full_set = []
test_comparison = []

p_c0 = 0.5
p_c1 = 0.5
pred_class = []
j = 0


with open('train.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

with open('test.csv', newline='') as csvfile:
    test = list(csv.reader(csvfile))

for item in data:
    full_set.append([float(item[0]), float(item[1])])
    if item[2] == '0':
        class0.append([float(item[0]), float(item[1])])
    else:
        class1.append([float(item[0]), float(item[1])])

for item in test:
    test_full_set.append([float(item[0]), float(item[1])])
    test_comparison.append(int(item[2]))

mean = d.mean_data(full_set, 0, 1)
mean_c0 = d.mean_data(class0, 0, 1)
mean_c1 = d.mean_data(class1, 0, 1)

cov = d.covariance2D(mean, full_set)
cov_c0 = d.covariance2D(mean_c0, class0)
cov_c1 = d.covariance2D(mean_c1, class1)

for line in test_full_set:
    px_c0 = d.gaussian_pX(np.array(line), np.array(cov_c0), np.array(mean_c0))
    px_c1 = d.gaussian_pX(np.array(line), np.array(cov_c1), np.array(mean_c1))
    if px_c1 > px_c0:
        pred_class.append(1)
    else:
        pred_class.append(0)

for i in range(len(pred_class)):
    if pred_class[i] == test_comparison[i]:
        j = j + 1

print(pred_class)
print('Success rate =', j/400)
print('# of predictions correct =', j)

stop = timeit.default_timer()
print('Time:', stop - start)

test_class0x = []
test_class1x = []
test_class0y = []
test_class1y = []

for row in range(len(test_full_set)):
    if pred_class[row] == 1:
        test_class1x.append(test_full_set[row][0])
        test_class1y.append(test_full_set[row][1])
    else:
        test_class0x.append(test_full_set[row][0])
        test_class0y.append(test_full_set[row][1])

plt.scatter(test_class0x, test_class0y, c='red')
plt.scatter(test_class1x, test_class1y, c='blue')
plt.show()

