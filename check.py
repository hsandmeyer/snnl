#!/usr/bin/env python3

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dense(x):
    W = np.array([[1, -1], [-1, 2]])
    B = np.array([-2.5, 2.5])
    return np.dot(x, W) + B


input_1 = np.array([[1.0, 2.0], [3.0, 4.0]])

input_2 = np.array([[3.141, 1.414], [0.0, 42.0]])


tmp_1_0 = dense(input_1)
tmp_1_0 = sigmoid(tmp_1_0)

tmp_1_1 = dense(tmp_1_0)
tmp_1_1 = sigmoid(tmp_1_1)

tmp_2_0 = dense(input_2)
tmp_2_0 = sigmoid(tmp_2_0)

tmp_1_3 = tmp_1_1 + tmp_1_0
tmp_1_4 = tmp_1_3 + tmp_1_0
combined = tmp_1_4 + tmp_2_0

res = np.sum(combined)
print(res)
