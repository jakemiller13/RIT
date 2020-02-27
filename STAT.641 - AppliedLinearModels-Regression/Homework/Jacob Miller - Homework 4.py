#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 06:19:15 2020

@author: Jake
"""

import pandas as pd
import patsy
import statsmodels.api as sm
import matplotlib.pyplot as plt

#############
# Problem 2 #
#############
df = pd.DataFrame(data = {'Pressure_(x)': [30, 31, 32, 33, 34, 35, 36],
                          'Mileage_(y1)' : [29.5, 32.1, 36.3, 38.2,
                                            37.7, 33.6, 26.8],
                          'Mileage_(y2)' : [30.2, 34.5, 35.0, 37.6,
                                            36.1, 34.2, 27.4]}).\
                          set_index('Pressure_(x)')

###############
# Problem 2.a #
###############
df = df[df.columns[0]].append(df[df.columns[1]])
plt.scatter(x = df.index.values, y = df)
plt.show()
print('Optimal tire pressure appears to be between 32 - 34 psi')

###############
# Problem 2.b #
###############
#y, X = patsy.dmatrices('y ~ {} + {}'.format(*df.columns[1:]), df)
##y, X = patsy.dmatrices('y ~ Mileage_(y1) + Mileage (y2)', df)
#
#model = sm.OLS()