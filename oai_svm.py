import numpy as np
import os

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import sys

n_splits=10

oai_coef_file = ###'./oai_coeff_file/'
coef_dir = os.listdir(oai_coef_file)
coef_num = len(coef_dir)

file_path = ###'./oai_label.txt'

label_dict = {}

oai_data = []
oai_label = []

with open(file_path, 'r') as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) == 2: 
            number, label = parts
            label_dict[number] = int(label)  

for i in range(coef_num):
    npy_name = coef_dir[i]
    npy_data = np.load(oai_coef_file + npy_name)
    npy_data = npy_data[0]
    #npy_data = np.concatenate((npy_data[0], npy_data[1])) 
    oai_data.append(npy_data)
    npy_name = npy_name.split('.')[0]
    npy_label = label_dict[npy_name]
    oai_label.append(npy_label)

x = np.array(oai_data).astype(float)
y = np.array(oai_label).astype(float)
print(x.shape)


train_sizes = np.round(np.arange(0.9, 0., -0.1), 1)
number_shapes = np.floor(coef_num * train_sizes)
number_shapes //= 2
number_shapes *= 2

train_sizes = number_shapes / coef_num
#clf = SVC(kernel='rbf', C=200)#, random_state=42) # C = 2.5
clf = SVC(kernel="linear", random_state=0, tol=1e-5, max_iter=4000)
results = {}
iter_num = 5000
result_mat = np.zeros((9, iter_num))

for ti in range(iter_num):
    i = 0
    for train_size in train_sizes:
        monte_carlo_split = StratifiedShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=None)
        accurracy = np.zeros(n_splits)
        for n, (train_index, test_index) in enumerate(monte_carlo_split.split(x, y)):
            x_train = x[train_index]
            x_test = x[test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            clf = clf.fit(x_train, y_train)
            accurracy[n] = clf.score(x_test, y_test)
        result_mat[i, ti] = np.mean(accurracy)
        i += 1

print("mean of vec: {}".format(np.mean(result_mat, axis=1)))
print("max of vec: {}".format(np.max(result_mat, axis=1)))
print("min of vec: {}".format(np.min(result_mat, axis=1)))





