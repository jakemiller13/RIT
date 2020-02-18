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
#from statsmodels.tools import add_constant
import statsmodels.formula.api as smf
#import scipy.stats as stats

df = pd.read_excel('Data/data-table-B1.xlsx')
df = df.rename(columns = {'y': 'Games_won',
                          'x1': 'Rushing_yards',
                          'x2': 'Passing_yards',
                          'x3': 'Punting_average',
                          'x4': 'Field_goal_percentage',
                          'x5': 'Turnover_differential',
                          'x6': 'Penalty_yards',
                          'x7': 'Percent_rushing',
#                          'x8': 'Opponents\' rushing yards',
                          'x8': 'Opponent_rushing_yards',
                          'x9': 'Opponent_passing_yards'})

def title_print(text):
    '''
    Used throughout to print section titles
    '''
    text_len = len(text)
    print()
    print('#' * (text_len + 4))
    print('|', text, '|')
    print('#' * (text_len + 4))

################
# Problem 3.1a #
################
title_print('Problem 3.1a')
y, X = patsy.dmatrices('y ~ Passing_yards + Percent_rushing + \
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

# Note statsmodels prints out ANOVA for each regressor
aov_table = sm.stats.anova_lm(results, typ = 1)

# Calculate F test
MS_R = aov_table['mean_sq'][:-1].sum() / aov_table['df'][:-1].sum()
MS_Res = aov_table['mean_sq'][-1]
F = MS_R / MS_Res
p = 1 - scipy.stats.f.cdf(F, aov_table['df'][:-1].sum(), aov_table['df'][-1])

print(results.summary())
print('\n--- Analysis of Variance table ---\n{}'.format(aov_table))
print('\nRegression F: {}'.format(round(F, 2)))
print('Regression p: {}'.format(round(p, 4)))
print('--> Regression is significant <--')

################
# Problem 3.1c #
################
title_print('Problem 3.1c')
t_stat = -scipy.stats.t.ppf(0.025, len(X) - 2)
t_values = abs(np.round(results.tvalues[1:], 3))
p_values = np.round(results.pvalues[1:], 3)
print('t-statistic: {}'.format(round(t_stat, 3)))
print('B2: abs(t_0) = {} | p = {}'.format(t_values[0], p_values[0]))
print('B7: abs(t_0) = {} | p = {}'.format(t_values[1], p_values[1]))
print('B8: abs(t_0) = {} | p = {}'.format(t_values[2], p_values[2]))
print('--> abs(t_0) > t-statistic, so all are significant <--')