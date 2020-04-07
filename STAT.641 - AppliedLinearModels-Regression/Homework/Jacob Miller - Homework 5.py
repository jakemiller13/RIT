#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 09:31:19 2020

@author: Jake
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import patsy
import statsmodels.api as sm
from itertools import combinations, compress

def title_print(text):
    '''
    Used throughout to print section titles
    '''
    print()
    print('#' * (len(text) + 4))
    print('|', text, '|')
    print('#' * (len(text) + 4))

###############
# Problem 5.1 #
###############
df = pd.DataFrame(data = {'Temperature' : [24.9, 35.0, 44.9, 55.1,
                                           65.2, 75.2, 85.2, 95.2],
                          'Viscosity' : [1.133, 0.9772, 0.8532, 0.7550,
                                         0.6723, 0.6021, 0.5420, 0.5074]})
#################
# Problem 5.1.a #
#################
title_print('Problem 5.1.a')
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
title_print('Problem 5.1.b')
y, X = patsy.dmatrices('Viscosity ~ Temperature', df)
model = sm.OLS(y, X)
results = model.fit()
results.model.data.design_info = X.design_info
print(results.summary())

# Get residuals and probability for plot
residuals = results.resid
Prob = [(i - 1/2) / len(y) for i in range(len(y))]

# Plot residuals vs. fitted values
fig, ax = plt.subplots(figsize = (8, 8))
ax.scatter(results.fittedvalues, residuals)
ax.axhline(0)
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
plt.title('Residuals Versus Predicted Response')
plt.show()

# Calculate OLS using residuals to plot straight line. Get y values from model
resid_results = sm.OLS(Prob, sm.add_constant(sorted(residuals))).fit()
X_range = np.linspace(min(residuals), max(residuals), len(residuals))

# Normality plot
fig = plt.figure(figsize = (8, 8))
plt.scatter(sorted(residuals), Prob)
plt.plot(X_range, resid_results.params[0] + resid_results.params[1] * X_range)
plt.xlabel('Residual')
plt.ylabel('Probability')
plt.title('Normal Probability Plot')
plt.show()

print('\n--> Clear non-linearity in residual plot <--')
print('--> Normality appears to have problems <--')

#################
# Problem 5.1.c #
#################
title_print('Problem 5.1.c')
y, X = patsy.dmatrices('Viscosity ~ np.log(Temperature)', df)
model = sm.OLS(y, X)
results = model.fit()
results.model.data.design_info = X.design_info
print(results.summary())

# Get residuals and probability for plot
residuals = results.resid
Prob = [(i - 1/2) / len(y) for i in range(len(y))]

# Plot residuals vs. fitted values
fig, ax = plt.subplots(figsize = (8, 8))
ax.scatter(results.fittedvalues, residuals)
ax.axhline(0)
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
plt.title('Residuals Versus Predicted Response')
plt.show()

# Calculate OLS using residuals to plot straight line. Get y values from model
resid_results = sm.OLS(Prob, sm.add_constant(sorted(residuals))).fit()
X_range = np.linspace(min(residuals), max(residuals), len(residuals))

# Normality plot
fig = plt.figure(figsize = (8, 8))
plt.scatter(sorted(residuals), Prob)
plt.plot(X_range, resid_results.params[0] + resid_results.params[1] * X_range)
plt.xlabel('Residual')
plt.ylabel('Probability')
plt.title('Normal Probability Plot')
plt.show()

print('\n--> Residual vs. Response plot mildly improved <--')
print('--> R-squared value slightly increased <--')
print('--> Overall improvement seems minimal <--')

###############
# Problem 6.1 #
###############
title_print('Problem 6.1')

# df = pd.read_excel('Data/data-table-B2.xlsx')
# y, X = patsy.dmatrices('y ~ x1 + x2 + x3 + x4 + x5', df)

def run_analysis(drop_point = None):
    '''
    Parameters
    ----------
    drop_point : int or list-like, optional
        Point(s) to drop for analysis. The default is None.

    Returns
    -------
    Points to potentially drop, if drop_point == None.

    '''
    df = pd.read_excel('Data/data-table-b2.xlsx')
    df.columns = ['y', 'x1', 'x2', 'x3', 'x4', 'x5']
    
    # Drop influential points, if necessary
    if drop_point:
        df = df.drop(np.array(drop_point) - 1)
    y, X = patsy.dmatrices('y ~ x1 + x2 + x3 + x4 + x5', df)

    # Fit model, get influence statistics
    model = sm.OLS(y, X)
    results = model.fit()
    results.model.data.design_info = X.design_info
    
    # If dropping points, only run to here and then exit function
    if drop_point:
        print('\nPoints dropped: {}'.format(drop_point))
        print('Coefficients: {}'.format(np.round(results.params, 3)))
        print('R-squared: {}'.format(round(results.rsquared, 3)))
        return
    else:
        print('Coefficients: {}'.format(np.round(results.params, 3)))
        print('R-squared: {}\n'.format(round(results.rsquared, 3)))
    
    infl = results.get_influence()
    infl_df = infl.summary_frame()
    print(infl_df.to_string())

    # Hat diagonals
    n, p = df.shape
    lev_pt = 2 * p / n
    dhat_pts = list(infl_df[infl_df['hat_diag'] > lev_pt].index + 1)
    print('\n***| Hat Diagonal |***')
    print('Leverage calculation (2 * p \ n) = {}'.format(round(lev_pt, 3)))
    print('Points where hat diagonal exceeds leverage calculation: {}'.
          format(dhat_pts))
    
    # Cook's D
    print('\n***| Cook\'s D |***')
    print('Points where Cook\'s D is > 1: {}'.
      format(list(infl_df[infl_df['cooks_d'] > 1].index + 1)))
    
    # DFFITS
    print('\n***| DFFITS |***')
    DFFITS_cutoff = 2 * np.sqrt(p / n)
    print('Points which exceed DFFITS cutoff: {}'.
          format(list(infl_df[infl_df['dffits'] > DFFITS_cutoff].index + 1)))
    
    # DFBETAS
    print('\n***| DFBETAS |***')
    DFBETAS_cutoff = 2 / np.sqrt(n)
    for col in infl_df.columns:
        if 'dfb' in col:
            print('Points which exceed DFBETAS cutoff for {}: {}'.
                  format(col,
                         list(infl_df[infl_df[col] > DFBETAS_cutoff].
                              index + 1)))
    
    # COVRATIO
    print('\n***| COVRATIO |***')
    COVRATIO_cutoff_pos = 1 + 3 * p / n
    COVRATIO_cutoff_neg = 1 - 3 * p / n
    gt_cutoff = np.array(list(compress(range(len(infl.cov_ratio)),
                              infl.cov_ratio > COVRATIO_cutoff_pos))) + 1
    lt_cutoff = np.array(list(compress(range(len(infl.cov_ratio)),
                              infl.cov_ratio < COVRATIO_cutoff_neg))) + 1
    print('Points which are greater than COVRATIO upper bound cutoff: {}'.
          format(gt_cutoff))
    print('Points which are less than COVRATIO lower bound cutoff: {}'.
          format(lt_cutoff))
    
    return dhat_pts

# All points
leverage_points = run_analysis(drop_point = None)

# Drop influential points
for i in range(1, len(leverage_points) + 1):
    comb = combinations(leverage_points, i)
    for pts in comb:
        run_analysis(pts)

# Points 1, 4 appear influential
print('\n--> Points 1 and 4 appear influential <--')
