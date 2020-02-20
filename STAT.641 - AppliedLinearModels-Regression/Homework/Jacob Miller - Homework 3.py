#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:23:32 2020

@author: Jake
"""

import pandas as pd
import numpy as np
import patsy
import scipy
import statsmodels.api as sm
from astropy.table import Table
from sympy import symbols

def title_print(text):
    '''
    Used throughout to print section titles
    '''
    text_len = len(text)
    print()
    print('#' * (text_len + 4))
    print('|', text, '|')
    print('#' * (text_len + 4))

###############
# Problem 3.1 #
###############
df = pd.read_excel('Data/data-table-B1.xlsx')
df = df.rename(columns = {'y': 'Games_won',
                          'x1': 'Rushing_yards',
                          'x2': 'Passing_yards',
                          'x3': 'Punting_average',
                          'x4': 'Field_goal_percentage',
                          'x5': 'Turnover_differential',
                          'x6': 'Penalty_yards',
                          'x7': 'Percent_rushing',
                          'x8': 'Opponent_rushing_yards',
                          'x9': 'Opponent_passing_yards'})

################
# Problem 3.1a #
################
title_print('Problem 3.1a')
y, X = patsy.dmatrices('Games_won ~ Passing_yards + Percent_rushing + \
                       Opponent_rushing_yards', df)

parens = np.matmul(X.T, X)
Xs = np.matmul(np.linalg.inv(parens), X.T)
b_hat = np.round(np.matmul(Xs, y), 4)

print('y_hat = {} + {} * x_2 + {} * x_7 + {} * x_8'.format(b_hat[0],
                                                           b_hat[1],
                                                           b_hat[2],
                                                           b_hat[3]))

################
# Problem 3.1b #
################
title_print('Problem 3.1b')
results = sm.OLS(y, X).fit()
results.model.data.design_info = X.design_info

# Note statsmodels prints out ANOVA for each individual regressor
aov_table = sm.stats.anova_lm(results, typ = 1)

print(results.summary())
print('\n--- Analysis of Variance table ---\n{}'.format(aov_table))
print('\nRegression F: {}'.format(round(results.fvalue, 2)))
print('Regression p: {}'.format(round(results.f_pvalue, 4)))
print('\n--> Regression is significant <--')

################
# Problem 3.1c #
################
title_print('Problem 3.1c')
t_stat = -scipy.stats.t.ppf(0.025, len(X) - 2)
t_values = abs(np.round(results.tvalues[1:], 3))
p_values = np.round(results.pvalues[1:], 3)
table = Table([['B2', 'B7', 'B8'], t_values, p_values],
              names = ('Coef', 't_0', 'p-value'))

print(table)
print('\nt-statistic = {}'.format(round(t_stat, 3)))
print('\n--> abs(t_0) > t-statistic, so all are significant <--')

################
# Problem 3.1d #
################
title_print('Problem 3.1d')
print('R^2 = {}%'.format(round(100 * results.rsquared, 2)))
print('Adj-R^2 = {}%'.format(round(100 * results.rsquared_adj, 2)))

################
# Problem 3.1e #
################
title_print('Problem 3.1e')
y2, X2 = patsy.dmatrices('y ~ Passing_yards + Opponent_rushing_yards', df)

parens2 = np.matmul(X2.T, X2)
Xs2 = np.matmul(np.linalg.inv(parens2), X2.T)
b_hat2 = np.round(np.matmul(Xs2, y2), 4)

results2 = sm.OLS(y2, X2).fit()
results2.model.data.design_info = X2.design_info

partial_F = round((results.ess - results2.ess) / results.mse_resid, 2)

print('reduced y_hat = {} + {} * x_2 + {} * x_8'.format(b_hat2[0],
                                                        b_hat2[1],
                                                        b_hat2[2]))
print('partial F = {}'.format(partial_F))
print('\n--> {} < {}, therefore B7 is significant <--'.format(partial_F,
      round(results.fvalue, 2)))

################
# Problem 3.10 #
################
df = pd.read_excel('Data/data-table-B11.xlsx')

#################
# Problem 3.10a #
#################
title_print('Problem 3.10a')
y, X = patsy.dmatrices('Quality ~ Clarity + Aroma + Body + Flavor + \
                       Oakiness', df)

parens = np.matmul(X.T, X)
Xs = np.matmul(np.linalg.inv(parens), X.T)
b_hat = np.round(np.matmul(Xs, y), 4)

print('y_hat = {} + {} * x_1 + {} * x_2 + {} * x_3 + {} * x_4 + {} * x_5'.\
      format(b_hat[0], b_hat[1], b_hat[2], b_hat[3], b_hat[4], b_hat[5]))

#################
# Problem 3.10b #
#################
title_print('Problem 3.10b')
results = sm.OLS(y, X).fit()
results.model.data.design_info = X.design_info

aov_table = sm.stats.anova_lm(results, typ = 1)

print(results.summary())
print('\n--- Analysis of Variance table ---\n{}'.format(aov_table))
print('\nRegression F: {}'.format(round(results.fvalue, 2)))
print('Regression p: {}'.format(round(results.f_pvalue, 4)))
print('\n--> Regression is significant <--')

#################
# Problem 3.10c #
#################
title_print('Problem 3.10c')
t_stat = -scipy.stats.t.ppf(0.025, len(X) - 2)
t_values = abs(np.round(results.tvalues[1:], 3))
p_values = np.round(results.pvalues[1:], 3)
table = Table([['B1', 'B2', 'B3', 'B4', 'B5'], t_values, p_values],
              names = ('Coef', 't_0', 'p-value'))

print(table)
print('\nt-statistic = {}'.format(round(t_stat, 3)))
print('\n--> B4, B5: abs(t_0) > t-statistic so these are significant <--')

#################
# Problem 3.10d #
#################
title_print('Problem 3.10d')
y2, X2 = patsy.dmatrices('Quality ~ Aroma + Flavor', df)

parens = np.matmul(X2.T, X2)
Xs2 = np.matmul(np.linalg.inv(parens), X2.T)
b_hat2 = np.round(np.matmul(Xs2, y2), 4)
results2 = sm.OLS(y2, X2).fit()
results2.model.data.design_info = X2.design_info

table = Table([['R^2', 'Adj-R^2'],
               [round(100 * results.rsquared, 2),
                round(100 * results.rsquared_adj, 2)],
               [round(100 * results2.rsquared, 2),
                round(100 * results2.rsquared_adj, 2)]],
               names = (' ', 'Full model', 'Reduced model'))
print(table)
print('\n--> Very similar, so models are similar <--')

#################
# Problem 3.10e #
#################
title_print('Problem 3.10e')
ci_1 = np.round(results.conf_int()[4], 3)
ci_2 = np.round(results2.conf_int()[2], 3)

print('Full model: {} to {}'.format(ci_1[0], ci_1[1]))
print('Reduced model: {} to {}'.format(ci_2[0], ci_2[1]))
print('\n--> Very similar again, so similar models <--')

#################
# Problem 3.25a # Note: this is problem 3.21 in 4th edition
#################
title_print('Problem 3.25a')
b, b0, b1, b2, b3, b4 = symbols('b b0 b1 b2 b3 b4')
y, x1, x2, x3, x4, eps = symbols('y x1 x2 x3 x4 eps')
gamma_0, gamma_1, z = symbols('gamma_0 gamma_1 z')

beta = np.array([[b0], [b1], [b2], [b3], [b4]])
X = np.array([1, x1, x2, x3, x4])
y = np.matmul(X, beta) + eps

# H0: b1 = b2 = b3 = b4
beta2 = np.array([[b0], [b1], [b1], [b1], [b1]])
y2 = np.matmul(X, beta2) + eps

T = np.array([[0, 1, -1, 0, 0],
              [0, 0, 1, -1, 0],
              [0, 0, 0, 1, -1]])
c = np.array([[0], [b], [b], [b], [b]])

print('y = {}'.format(y))
print('H0: b1 = b2 = b3 = b4 = b')
print('\nTherefore: b1 - b2 = 0, b2 - b3 = 0, b3 - b4 = 0')
print('\nT = \n{}\n\nbeta = \n{}\n\nc = \n{}'.format(T, beta, c))
print('\ny = {}'.format(y2))
print('\nWhere:\ngamma_0 = b0\ngamma_1 = b\nz = x1 + x2 + x3 + x4')
print('\n--> Reduced model: y = gamma_0 + gamma_1 * z + eps <--')

#################
# Problem 3.25b #
#################
title_print('Problem 3.25b')

beta = np.array([[b0], [b1], [b2], [b3], [b4]])
X = np.array([1, x1, x2, x3, x4])
y = np.matmul(X, beta) + eps

# H0: b1 = b2, b3 = b4
beta2 = np.array([[b0], [b1], [b1], [b3], [b3]])
y2 = np.matmul(X, beta2) + eps

T = np.array([[0, 1, -1, 0, 0],
              [0, 0, 0, 1, -1]])
c = np.array([[0], [0]])

print('y = {}'.format(y))
print('H0: b1 = b2, b3 = b4')
print('\nTherefore: b1 - b2 = 0, b3 - b4 = 0')
print('\nT = \n{}\n\nbeta = \n{}\n\nc = \n{}'.format(T, beta, c))
print('\ny = {}'.format(y2))
print('\nWhere:\ngamma_0 = b0\ngamma_1 = b1\ngamma_3 = b3')
print('z1 = x1 + x2\nz3 = x3 + x4')
print('\n--> Reduced model: y = gamma_0 + gamma_1 * z1 + gamma_3 * z3 <--')