#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 11:17:49 2020

@author: Jake
"""

import numpy as np
import pandas as pd
from scipy import stats

def title_print(text):
    '''
    Used throughout to print section titles
    '''
    text_len = len(text)
    print()
    print('#' * (text_len + 4))
    print('#', text, '#')
    print('#' * (text_len + 4))

#################
# Class example #
#################
title_print('Class example')
Y = np.array([[166], [163], [151], [177], [163]])
X = np.array([[12], [12.5], [10], [14], [11]])
X = np.concatenate((np.ones(X.shape), X), axis = 1)

inv = np.linalg.inv(np.matmul(X.T, X))
b_hat = np.matmul(np.matmul(inv, X.T), Y)

X_h = 10.5
y_hat = b_hat[0] + b_hat[1] * X_h

print('b_hat\n', b_hat)
print('\ny_hat\n', y_hat)

###############
# Problem 1.a #
###############
title_print('Problem 1.a')
X = np.array([[-8], [-4], [0], [4], [8]])
X = np.concatenate((np.ones(X.shape), X), axis = 1)
Y = np.array([[11.7], [11], [10.2], [9], [7.8]])

inv = np.linalg.inv(np.matmul(X.T, X))
b_hat = np.matmul(np.matmul(inv, X.T), Y)

print(b_hat)

###############
# Problem 1.c #
###############
title_print('Problem 1.c')
X_h = -6
y_hat = b_hat[0] + b_hat[1] * X_h

print(y_hat)

###############
# Problem 1.e #
###############
title_print('Problem 1.e')

# Constants
alpha = 0.05
t_stat = stats.t.ppf(alpha / 2, len(Y) - 2)

# Calculate SS_T
sum_y_squared = sum(Y ** 2)
squared_sum_y = sum(Y) ** 2
SS_T = sum_y_squared - squared_sum_y / len(Y)

# Calculate S_XY
sum_xy = sum(X[:,1].reshape(-1, 1) * Y)
sum_x_sum_y_over_n = sum(X[:, 1]) * sum(Y) / len(Y)
S_XY = sum_xy - sum_x_sum_y_over_n

# Calculate SS_res
SS_res = SS_T - b_hat[1] * S_XY

# Calculate MS_res (variance)
MS_res = SS_res / (len(Y) - 2)

## Using example 2.7 on page 34. 
