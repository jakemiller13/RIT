#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 09:31:19 2020

@author: Jake
"""

import pandas as pd
import matplotlib.pyplot as plt
import patsy
import statsmodels.api as sm

df = pd.DataFrame(data = {'Temperature' : [24.9, 35.0, 44.9, 55.1,
                                           65.2, 75.2, 85.2, 95.2],
                          'Viscosity' : [1.133, 0.9772, 0.8532, 0.7550,
                                         0.6723, 0.6021, 0.5420, 0.5074]})
#################
# Problem 5.1.a #
#################
fig = plt.figure(figsize = (8, 8))
plt.scatter(df['Temperature'], df['Viscosity'])
plt.xlabel('Temperature (deg. C)')
plt.ylabel('Viscosity (mPa * s)')
plt.title('Problem 5.1.a')
plt.text(50, 1, 'Straight-line model does\nNOT look adequate',
         fontdict = {'fontsize': 20})
plt.show()

#################
# Problem 5.1.b #
#################
y, X = patsy.dmatrices('Viscosity ~ Temperature', df)
model = sm.OLS(y, X)
results = model.fit()
results.model.data.design_info = X.design_info
print(results.summary())

# Get residuals and probability for plot
residuals = results.resid
Prob = [(i - 1/2) / len(y) for i in range(len(y))]
fig = plt.figure(figsize = (8, 8))
plt.scatter(sorted(residuals), Prob)
# plt.plot(X_range, resid_results.params[0] + resid_results.params[1] * X_range)
plt.xlabel('Residual')
plt.ylabel('Probability')
plt.title('Normal Probability Plot')
plt.show()