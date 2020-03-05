# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:28:37 2020

@author: jmiller
"""

import pandas as pd
import numpy as np
import patsy
import statsmodels.api as sm
import scipy
import matplotlib.pyplot as plt
from astropy.table import Table

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

table = Table([['Regression', 'Residual', 'Total'], # Source of Variation
               [SS_reg, '[  ]', SS_tot], # Sum of Squares
               ['[  ]', '[  ]', '[  ]'], # Degrees of Freedom
               ['[  ]', '[  ]', '    '], # Mean Square
               ['[  ]', '    ', '    '], # F0
               ['[  ]', '    ', '    ']], # P-value
              names = ('Source of Variation', 'Sum of Squares',
                       'Degrees of Freedom', 'Mean Square', 'F0', 'P-value'))

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

# Final table
table = Table([['Regression', 'Residual', 'Total'], # Source of Variation
               [SS_reg, SS_res, SS_tot], # Sum of Squares
               [DoF_reg, DoF_res, DoF_tot], # Degrees of Freedom
               [MS_reg, MS_res, ''], # Mean Square
               [F0, '', ''], # F0
               [P, '', '']], # P-value
              names = ('Source of Variation', 'Sum of Squares',
                       'Degrees of Freedom', 'Mean Square', 'F0', 'P-value'))

print(table.to_pandas().to_string())
print('\n--> Small P-value [{:e}] | Reject null hypothesis <--'.format(P))

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
title_print('Problem 2.a')
df_com = df[df.columns[0]].append(df[df.columns[1]])
plt.scatter(x = df_com.index.values, y = df_com)
plt.xlabel('Pressure (psi)')
plt.ylabel('Milage (thousands)')
plt.show()
print('--> Optimal tire pressure appears to be between 32 - 34 psi <--')

###############
# Problem 2.b #
###############
title_print('Problem 2.b')
y, X = patsy.dmatrices('df_com.values ~ df_com.index', df_com)

model = sm.OLS(y, X)
results = model.fit()
results.model.data.design_info = X.design_info

print('--> y = {} + {} * x <--'.format(*np.round((results.params[0],
                                                  results.params[1]),
                                                  2)))

fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(results, 1, ax = ax)
ax.set_xlabel('Pressure (psi)')
ax.set_ylabel('Mileage')
plt.show()

###############
# Problem 2.c #
###############
title_print('Problem 2.c')

# Setup - because each X has exactly 2 y, can just append to dataframe
df['Mean'] = df.apply(lambda x: np.mean(x), axis = 1)
df['PE Sums'] = df.apply(lambda x: ((x['Mileage_(y1)'] - x['Mean']) ** 2 +
                                    (x['Mileage_(y2)'] - x['Mean']) ** 2),
                         axis = 1)

# Get beta-hat, start to calculate SS_reg
b_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)
first_term = np.matmul(np.matmul(b_hat.T, X.T), y)
second_term = sum(y)**2 / len(y)

# Sum of Squares
SS_tot = np.round(float(sum(y**2) - sum(y)**2 / len(y)), 3)
SS_reg = np.round(float(first_term - second_term), 3)
SS_res = np.round(float(SS_tot - SS_reg), 3)
SS_pe = np.round(float(sum(df['PE Sums'])), 3)
SS_lof = np.round(float(SS_res - SS_pe), 3)

# Degrees of Freedom
DoF_reg = 1
DoF_tot = len(df_com) - 1
DoF_res = DoF_tot - DoF_reg
DoF_pe = len(df) # because each X has exactly 2 y, this is just num of X's
DoF_lof = DoF_res - DoF_pe

# Mean Squares
MS_reg = np.round(float(SS_reg / DoF_reg), 3)
MS_res = np.round(float(SS_res / DoF_res), 3)
MS_pe = np.round(float(SS_pe / DoF_pe), 3)
MS_lof = np.round(float(SS_lof / DoF_lof), 3)

# F0
F0 = round(MS_lof / MS_pe, 3)

# P-value
P = round(1 - scipy.stats.f.cdf(F0, DoF_reg, DoF_res), 10)

table = Table([['Regression', 'Error', 'Lack of Fit', 'Pure Error', 'Total'],
               [SS_reg, SS_res, SS_lof, SS_pe, SS_tot], # Sum of Squares
               [DoF_reg, DoF_res, DoF_lof, DoF_pe, DoF_tot], # Degrees of Freedom
               [MS_reg, MS_res, MS_lof, MS_pe, ''], # Mean Square
               ['', '', F0, '', ''], # F0
               ['', '', P, '', '']], # P-value
              names = ('Source of Variation', 'Sum of Squares',
                       'Degrees of Freedom', 'Mean Square', 'F0', 'P-value'))

print(table.to_pandas().to_string())

###############
# Problem 2.d #
###############
title_print('Problem 2.d')
print('--> R = {} | Low R-squared suggests model does not fit data well <--'.\
      format(round(results.rsquared, 5)))

###############
# Problem 2.e #
###############
title_print('Problem 2.e')
print('--> P = {} | Reject hypothesis that model describes data <--'.format(P))

###############
# Problem 2.f #
###############
title_print('Problem 2.f')
print('--> Assumed first order linear regression, bad assumption <--')

###############
# Problem 4.3 #
###############
df = pd.read_excel('Data/data-table-B2.xlsx')
y, X = patsy.dmatrices('y ~ x4', df)
model = sm.OLS(y, X)
results = model.fit()
results.model.data.design_info = X.design_info

################
# Problem 4.3a #
################
title_print('Problem 4.3a')

# Get residuals and probability for plot
residuals = results.resid
Prob = [(i - 1/2) / len(y) for i in range(len(y))]

# Calculate OLS using residuals to plot straight line. Get y values from model
resid_results = sm.OLS(Prob, sm.add_constant(sorted(residuals))).fit()
X_range = np.linspace(min(residuals), max(residuals), len(residuals))

# Normal Probability Plot + straight line
fig, ax = plt.subplots()
ax.scatter(sorted(residuals), Prob)
ax.plot(X_range, resid_results.params[0] + resid_results.params[1] * X_range)
ax.set_xlabel('Residual')
ax.set_ylabel('Probability')
plt.title('Normal Probability Plot')
plt.show()

print('--> Minimal fluctuation suggests normality is ok <--')

################
# Problem 4.3b #
################
title_print('Problem 4.3b')

fig, ax = plt.subplots()
ax.scatter(results.fittedvalues, residuals)
ax.axhline(0)
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
plt.title('Residuals Versus Predicted Response')
plt.show()

print('--> Appears to be non-constant variance (funnel or double-bow) <--')

###############
# Problem 4.5 #
###############
df = pd.read_excel('Data/data-table-B4.xlsx')
y, X = patsy.dmatrices('y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9', df)
model = sm.OLS(y, X)
results = model.fit()
results.model.data.design_info = X.design_info

################
# Problem 4.5a #
################
title_print('Problem 4.5a')

# Get residuals and probability for plot
residuals = results.resid
Prob = [(i - 1/2) / len(y) for i in range(len(y))]

# Calculate OLS using residuals to plot straight line. Get y values from model
resid_results = sm.OLS(Prob, sm.add_constant(sorted(residuals))).fit()
X_range = np.linspace(min(residuals), max(residuals), len(residuals))

# Normal Probability Plot + straight line
fig, ax = plt.subplots()
ax.scatter(sorted(residuals), Prob)
ax.plot(X_range, resid_results.params[0] + resid_results.params[1] * X_range)
ax.set_xlabel('Residual')
ax.set_ylabel('Probability')
plt.title('Normal Probability Plot')
plt.show()

print('--> Minimal fluctuation suggests normality is ok <--')

################
# Problem 4.5b #
################
title_print('Problem 4.5b')

fig, ax = plt.subplots()
ax.scatter(results.fittedvalues, residuals)
ax.axhline(0)
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
plt.title('Residuals Versus Predicted Response')
plt.show()

print('--> If outlier in top left corner is removed, plot has incline <--')
print('--> from lower left to top right. Otherwise no significant pattern <--')

################
# Problem 4.5c #
################
title_print('Problem 4.5c')

fig, ax = plt.subplots(figsize = (10, 20))
fig = sm.graphics.plot_partregress_grid(results, fig = fig)
plt.show()
print('--> Intercept, x1 have large influence. x2 minimal influence. <--')
print('--> x3 and above appear to not have any influence <--')

################
# Problem 4.5d #
################
title_print('Problem 4.5d')

infl = results.get_influence()
print(infl.summary_table())
print('--> "Obs 15" (i.e. observation 16 because table is 0-based) <--')
print('--> has larger absolute value than others, likely outlier <--')

################
# Problem 4.17 #
################
df = pd.read_excel('Data/data-table-B10.xlsx')
y, X = patsy.dmatrices('y ~ x1 + x2', df)
model = sm.OLS(y, X)
results = model.fit()
results.model.data.design_info = X.design_info

#################
# Problem 4.17a #
#################
title_print('Problem 4.17a')

# Get residuals and probability for plot
residuals = results.resid
Prob = [(i - 1/2) / len(y) for i in range(len(y))]

# Calculate OLS using residuals to plot straight line. Get y values from model
resid_results = sm.OLS(Prob, sm.add_constant(sorted(residuals))).fit()
X_range = np.linspace(min(residuals), max(residuals), len(residuals))

# Normal Probability Plot + straight line
fig, ax = plt.subplots()
ax.scatter(sorted(residuals), Prob)
ax.plot(X_range, resid_results.params[0] + resid_results.params[1] * X_range)
ax.set_xlabel('Residual')
ax.set_ylabel('Probability')
ax.set_ylim(0, 1)
plt.title('Normal Probability Plot')
plt.show()

print('--> Normality plot exhibits negative skew, therefore problems <--')

#################
# Problem 4.17b #
#################
title_print('Problem 4.17b')

fig, ax = plt.subplots()
ax.scatter(results.fittedvalues, residuals)
ax.axhline(0)
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
plt.title('Residuals Versus Predicted Response')
plt.show()

print('--> Definite non-linear pattern for residual vs. predicted plot <--')

#################
# Problem 4.17c #
#################
title_print('Problem 4.17c')

y2, X2 = patsy.dmatrices('y ~ x2', df)
model2 = sm.OLS(y2, X2)
results2 = model2.fit()
results2.model.data.design_info = X2.design_info

infl1 = results.get_influence()
infl2 = results2.get_influence()
press1 = infl1.resid_press
press2 = infl2.resid_press
r_squared_pred1 = 1 - \
    infl1.ess_press/sm.stats.anova_lm(results, typ = 1).sum_sq.sum()
r_squared_pred2 = 1 - \
    infl2.ess_press/sm.stats.anova_lm(results2, typ = 1).sum_sq.sum()

comp_press = pd.DataFrame({'Ordinary Residuals': residuals,
                           'Full Model (PRESS_1)': press1,
                           'Viscosity to Temp Only (PRESS_2)': press2})
print(comp_press.to_string())

print('Full Model: PRESS Statistic = {} | R^2 (Pred) = {}%'.\
      format(round(infl1.ess_press, 3),
             round(100 * r_squared_pred1, 2)))
print('Partial Model: PRESS Statistic = {} | R^2 (Pred) = {}%'.\
      format(round(infl2.ess_press, 3),
             round(100 * r_squared_pred2, 2)))
print('--> Full Model: Lower PRESS, higher R^2 implies better prediction <--')