# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 07:45:29 2020

@author: jmiller
"""

import numpy as np
import pandas as pd
import patsy
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from itertools import compress

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

#############
# Problem 7 #
#############
title_print('Problem 7')

df = pd.read_csv('condo.txt',
                 header = None,
                 names = ['sale_price', 'floor', 'elevator_distance',
                          'ocean_view', 'end_unit', 'furnished'],
                 delimiter = '\t')

# Part 1
y, X = patsy.dmatrices('sale_price ~ floor + \
                                     elevator_distance + \
                                     ocean_view + \
                                     end_unit + furnished',
                                     df)

model = sm.OLS(y, X)
results = model.fit()
results.model.data.design_info = X.design_info

print('\n--- 7.1 ---')
print(results.summary())
print('\n-->> Model: y = {} + {} * x1 + {} * x2 + '\
      '{} * x3 + {} * x4 + {} * x5 <<--'.\
      format(*np.round((results.params[0],
                        results.params[1],
                        results.params[2],
                        results.params[3],
                        results.params[4],
                        results.params[5]),
                       3)))


# Part 2
print('\n--- 7.2 ---')
sns.pairplot(df)
plt.show()

print(df.corr().to_string())
sns.heatmap(df.corr())
plt.show()
# Ocean view most correlated, followed by elevator distance

# Part 3
print('\n--- 7.3 ---')
print('R-squared: {}'.format(round(results.rsquared, 4)))
# relatively low r-squared

# Part 4
pvals = np.round(results.pvalues, 4)
print('\n--- 7.4 ---')
[print('x{} | p: {}'.format(i, j)) for i, j in enumerate(pvals)]

# Part 5
vif = np.round([variance_inflation_factor(X, i)
                for i in range(X.shape[1])], 4)
print('\n--- 7.5 ---')
[print('VIF_{}: {}'.format(i, vif[i])) for i, v in enumerate(vif)]

# Part 6
print('\n--- 7.6 ---')
corrs = np.abs(df.corr()['sale_price']).sort_values(ascending = False)[1:]

# Forward Selection
print('-> Forward Selection <-')
alpha_in = 0.25
t_in = round(-scipy.stats.t.ppf(alpha_in/2, len(X) - 2), 4)
print('alpha-to-enter: {}'.format(alpha_in))
print('t-statistic: {}'.format(t_in))

for i, j in enumerate(corrs, 1):
    to_include = list(corrs[:i].index)
    y, X = patsy.dmatrices('sale_price ~ {}'.\
                           format(' + '.join(to_include)), df)

    to_include.insert(0, 'constant')
    model = sm.OLS(y, X)
    results = model.fit()
    results.model.data.design_info = X.design_info
    print('\nAdding: {}'.format(to_include[-1]))
    print(pd.DataFrame(data = {'T_Values': results.tvalues,
                               'P_Values': results.pvalues},
                       index = to_include).T.to_string())

# Backward Elimination
print('\n-> Backward Elimination <-')
alpha_out = 0.1
t_out = round(-scipy.stats.t.ppf(alpha_out / 2, len(X) - 2), 4)
print('alpha-to-remove: {}'.format(alpha_out))
print('t-statistic: {}\n'.format(t_out))

to_include = list(corrs.index)

for i, j in enumerate(corrs, 1):
    y, X = patsy.dmatrices('sale_price ~ {}'.\
                           format(' + '.join(to_include)), df)
    
    model = sm.OLS(y, X)
    results = model.fit()
    results.model.data.design_info = X.design_info
    
    df_index = to_include.copy()
    df_index.insert(0, 'constant')
    partial_df = pd.DataFrame(data = {'T_Values': results.tvalues,
                                      'P_Values': results.pvalues},
                              index = df_index)
    print(partial_df.T.to_string())
    min_t_idx = partial_df['T_Values'].argmin()
    min_t_val = partial_df['T_Values'].iloc[min_t_idx]
    min_t_var = partial_df.index[min_t_idx]
    
    if min_t_val < t_out:
        print('Removing: {}\n'.format(min_t_var))
        to_include.remove(min_t_var)
    else:
        break

# Reduced Model
print('-> Reduced Model <-')
red_coef = np.round(results.params)
print('y = {} + {} * ocean_view + {} * elevator_distance'.format(
      red_coef[0], red_coef[1], red_coef[2]))

# Part 7
print('\n--- 7.7 ---')
vif = np.round([variance_inflation_factor(X, i)
                for i in range(X.shape[1])], 4)
[print('VIF_{}: {}'.format(i, vif[i])) for i, v in enumerate(vif)]

# Part 8
print('\n--- 7.8 ---')
infl = results.get_influence()
infl_df = infl.summary_frame()
print(infl_df.head())
print('...continued...')
infl_pts = {}

# Leverage Points - Hat Diagonal
n, p = X.shape[0], X.shape[1] - 1
lev_pt = 2 * p / n
dhat_pts = list(infl_df[infl_df['hat_diag'] > lev_pt].index)
print('\n***| Hat Diagonal |***')
print('Leverage cutoff (2 * p \ n) = {}'.format(round(lev_pt, 3)))
print('Points where hat diagonal exceeds leverage cutoff: {}'.
    format(dhat_pts))

# Cook's D
cook_pts = list(infl_df[infl_df['cooks_d'] > 1].index)
print('\n***| Cook\'s D |***')
print('Points where Cook\'s D is > 1: {}'.
  format(cook_pts))

# DFFITS
DFFITS_cutoff = 2 * np.sqrt(p / n)
DFFITS_pts = list(infl_df[infl_df['dffits'] > DFFITS_cutoff].index)
print('\n***| DFFITS |***')
print('DFFITS cutoff (2 * sqrt(p / n)) = {}'.
      format(round(DFFITS_cutoff, 3)))
print('Points which exceed DFFITS cutoff: {}'.
      format(DFFITS_pts))

# DFBETAS
print('\n***| DFBETAS |***')
DFBETAS_cutoff = 2 / np.sqrt(n)
DFBETAS_pts = []
print('DFBETAS cutoff (2 / sqrt(n)) = {}'.
      format(round(DFBETAS_cutoff, 3)))
for col in infl_df.columns:
    if 'dfb' in col:
        temp_dfbeta = list(infl_df[infl_df[col] > DFBETAS_cutoff].index)
        DFBETAS_pts.extend(temp_dfbeta)        
        print('Points which exceed DFBETAS cutoff for {}: {}'.
              format(col,
                     list(temp_dfbeta)))

# COVRATIO
print('\n***| COVRATIO |***')
COVRATIO_cutoff_pos = 1 + 3 * p / n
COVRATIO_cutoff_neg = 1 - 3 * p / n
gt_cutoff = list(compress(range(len(infl.cov_ratio)),
                          infl.cov_ratio > COVRATIO_cutoff_pos))
lt_cutoff = list(compress(range(len(infl.cov_ratio)),
                          infl.cov_ratio < COVRATIO_cutoff_neg))
COVRATIO_pts = gt_cutoff + lt_cutoff
print('Upper COVRATIO cutoff (1 + 3 * p / n) = {}'.
      format(np.round(COVRATIO_cutoff_pos, 3)))
print('Lower COVRATIO cutoff (1 - 3 * p / n) = {}'.
      format(np.round(COVRATIO_cutoff_neg, 3)))
print('Points which are greater than COVRATIO upper bound cutoff:\n{}'.
      format(gt_cutoff))
print('Points which are less than COVRATIO lower bound cutoff:\n{}'.
      format(lt_cutoff))

# Most influential points
for i in dhat_pts + cook_pts + DFFITS_pts + DFBETAS_pts + COVRATIO_pts:
    infl_pts[i] = infl_pts.get(i, 0) + 1
most_infl = [pt for pt in infl_pts
             if infl_pts[pt] == max(infl_pts.values())]
print('\n***| MOST INFLUENTIAL POINTS |***') #points in every cutoff
print(sorted(most_infl))

# Check most influential point(s)
print()
for i in most_infl:
    print(df.iloc[i])

# Part 9
print('\n--- 7.9 ---')
resid = results.resid
Prob = [(i - 1/2) / len(y) for i in range(len(y))]

# Plot residuals vs. fitted values
fig, ax = plt.subplots(figsize = (8, 8))
ax.scatter(results.fittedvalues, resid)
ax.scatter(results.fittedvalues[207], resid[207], c = 'red')
ax.axhline(0)
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
plt.title('Residuals Versus Predicted Response')
plt.show()

# Calculate OLS from resid to plot straight line. y values from model
resid_results = sm.OLS(Prob, sm.add_constant(sorted(resid))).fit()
X_range = np.linspace(min(resid),
                      max(resid),
                      len(resid))

# Normality plot
fig = plt.figure(figsize = (8, 8))
plt.scatter(sorted(resid), Prob)
plt.scatter(sorted(resid)[207], Prob[207], c = 'red')
plt.plot(X_range,
         resid_results.params[0] + resid_results.params[1] * X_range)
plt.xlabel('Residual')
plt.ylabel('Probability')
plt.title('Normal Probability Plot')
plt.show()
print('---> Heavy-tailed distribution <---')
# Pred response mildly cone shaped
# Normality mild heavy tailed distribution

# Part 10
print('--- 7.10 ---')
