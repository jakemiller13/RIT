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
print('--> Regression is significant <--')

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
print('--> abs(t_0) > t-statistic, so all are significant <--')

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
print('--> {} < {}, therefore B7 is significant <--'.format(partial_F,
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
print('--> Regression is significant <--')

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
print('--> B4, B5: abs(t_0) > t-statistic so these are significant <--')

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