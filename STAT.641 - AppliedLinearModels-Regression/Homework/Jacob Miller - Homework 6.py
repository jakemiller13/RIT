#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 15:21:41 2020

@author: Jake
"""

# Package imports
import pandas as pd
import numpy as np
import patsy
import statsmodels.api as sm

# Read data
df = pd.read_excel('Data/data-table-B3.xlsx')
y, X = patsy.dmatrices('y ~ x10 + x11', df)

# Create model
model = sm.OLS(y, X)
results = model.fit()
results.model.data.design_info = X.design_info
print(results.summary())

# Linear regression model
coef = np.round(results.params, 3)
print('\ny = {} + ({} * x10) + ({} * x11)'.format(coef[0], coef[1], coef[2]))
print('No. High p-value [{}], so transmission not significant\n'.\
      format(round(results.pvalues[2], 3)))

# Interaction
y, X = patsy.dmatrices('y ~ x10 + x11 + x10 * x11', df)

# Create model with interaction
model = sm.OLS(y, X)
results = model.fit()
results.model.data.design_info = X.design_info
print(results.summary())

# Linear regression model with interaction
coef = np.round(results.params, 3)
print('\ny = {} + ({} * x10) + ({} * x11) + ({} * x10 * x11)'.\
      format(coef[0], coef[1], coef[2], coef[3]))
print('\nFor automatic transmission, x11 = 1, therefore:')
print('y = {} + ({} * x10)'.format(round(coef[0] + coef[2], 3),
                                   round(coef[1] + coef[3], 3)))
print('\nFor manual transmission, x11 = 0, therefore:')
print('y = {} + ({} * x10)'.format(coef[0], coef[1]))

# For manual transmission, gasoline mileage decreases more quickly