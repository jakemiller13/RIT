# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 07:45:29 2020

@author: jmiller
"""

import numpy as np
import pandas as pd
import patsy
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def title_print(text):
    '''
    Used throughout to print section titles
    '''
    print()
    print('#' * (len(text) + 4))
    print('|', text, '|')
    print('#' * (len(text) + 4))

#############
# Problem 1 #
#############
title_print('Problem 1')

# Problem setup
SS_reg = 5550.8166
SS_tot = 5784.5426

table = pd.DataFrame({'Source of Variation':
                          ['Regression', 'Residual', 'Total',],
                      'Sum of Squares': [SS_reg, np.nan, SS_tot],
                      'Degrees of Freedom': [np.nan, np.nan, np.nan],
                      'Mean Square': [np.nan, np.nan, ''],
                      'F0': [np.nan, '', ''],
                      'P-value': [np.nan, '', '']})
print(table.to_string())

# Sum of Squares
SS_res = SS_tot - SS_reg

# Degrees of Freedom
DoF_reg = 2
DoF_tot = 25 - 1
DoF_res = DoF_tot - DoF_reg

# Mean Squares
MS_reg = SS_reg / DoF_reg
MS_res = SS_res / DoF_res

# F0
F0 = MS_reg / MS_res

# P-value
P = 1 - scipy.stats.f.cdf(F0, DoF_reg, DoF_res)

table = pd.DataFrame({'Source of Variation':
                          ['Regression', 'Residual', 'Total',],
                      'Sum of Squares': [SS_reg, SS_res, SS_tot],
                      'Degrees of Freedom': [DoF_reg, DoF_res, DoF_tot],
                      'Mean Square': [MS_reg, MS_res, ''],
                      'F0': [F0, '', ''],
                      'P-value': [P, '', '']})

print()
print(table.to_string())
print('\n--> Small P-value [{:e}] | Reject null hypothesis <--'.format(P))

#############
# Problem 2 #
#############
title_print('Problem 2')

print('\
a. High VIF for runs and runs batted in (RBI) implies multicollinearity.\
    Can try centering linear terms to reduce VIF.\
\n\n\
b. Best model is one that contains first 3 variables (RBI, contract, KO).\
    Cp does not improve dramatically (i.e. does not decrease) when adding\
    additional 4th variable. Additionally, R-squared barely improves with\
    addition of 4th variable. In order to keep model simple but accurate,\
    only use RBI, contract, KO in model.')

#############
# Problem 6 #
#############
title_print('Problem 6')

# Part 1
df = pd.read_csv('Annual_due.txt',
                 header = None,
                 names = ['y', 'X'],
                 delimiter = '   ',
                 engine = 'python')
X, y = df['X'], df['y']
logreg = LogisticRegression()
logreg.fit(X.values.reshape(-1, 1), y.values)
inter = np.round(logreg.intercept_[0], 4)
coef = np.round(logreg.coef_[0][0], 4)
print('\n--- 6.1 ---')
print('B0: {}'.format(inter))
print('B1: {}'.format(coef))
print('\nResponse Function:')
print('(exp({} + {} * x1) / [1 + exp({} + {} * x1)'.format(*(inter, coef) * 2))

# Part 2
print('\n--- 6.2 ---')
fig, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(X, logreg.predict_proba(X.values.reshape([-1, 1]))[:, 1])
ax.set_xlabel('Increased Dues')
ax.set_ylabel('Renewed (0 = Yes, 1 = No)')
ax.set_ylim(-0.1, 1.1)
ax.set_title('6.2: Scatter Plot + Response')
plt.show()

# Part 3
print('\n--- 6.3 ---')
print('exp(b1) = {}'.format(round(np.exp(logreg.coef_)[0][0], 4)))
print('For every $1 increase, {} increase in odds ratio of non-renewal'.
      format(round(np.exp(logreg.coef_)[0][0], 4)))

# Part 4
print('\n--- 6.4 ---')
print('Prob. of renewal if $40 increase: {}'.format(
    np.round(logreg.predict_proba([[40]])[0][1], 4)))

# Part 5
pi = 0.75
print('\n--- 6.5 ---')
print('Linear predictor function:')
print('ln(pi / (1 - pi)) = B0 + B1 * x1')
print('If pi = {}, x1 = --> {} <--'.format(pi,
        np.round((np.log(pi / (1 - pi)) - logreg.intercept_) / logreg.coef_,
        4)[0][0]))