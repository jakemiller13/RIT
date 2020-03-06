# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:36:09 2020

@author: jmiller
"""

import pandas as pd
import numpy as np
import patsy
import statsmodels.api as sm
import matplotlib.pyplot as plt
from astropy.table import Table

# Note: Problems are separated into functions for easier troubleshooting

def title_print(text):
    '''
    Used throughout to print section titles
    '''
    print()
    print('#' * (len(text) + 4))
    print('|', text, '|')
    print('#' * (len(text) + 4))

def Problem_1():
    df = pd.DataFrame(data =
                      {'Estimate':      [np.nan, 5.3036, 4.0336, -9.3153,
                                         0.5884],
                       'Std. Error':    [0.1960, 2.5316, 2.4796, 2.4657,
                                         2.2852],
                       't value':       [8.438, np.nan, 1.627, -3.778,
                                         0.257],
                       'Pr(>|t|)':      [3.57e-13, 0.038834, 0.107111,
                                         0.000276, 0.797373]},
                      index = ['Intercept', 'x1', 'x2', 'x3', 'x4'])
    return df

def Problem_3():
    df = pd.DataFrame(data = {'DoF':            [3, np.nan, 23],
                              'Sum of Squares': [np.nan, 61.44300, 689.2600],
                              'Mean Square':    [np.nan, np.nan, ''],
                              'F Value':        [np.nan, '', ''],
                              'PR > F':         [np.nan, '', '']})
    return df

def Problem_6():
    df = pd.DataFrame(data = {'X': [4.0, 4.5, 5.0, 5.5, # Grams of Product
                                    6.0, 6.5, 7.0],
                              'y': [32, 43, 45, 51, 53, 61, 62]}) # Sud height
    y, X = patsy.dmatrices('y ~ X', df)
    model = sm.OLS(y, X)
    results = model.fit()
    results.model.data.design_info = X.design_info
    
    title_print('Problem 6.a')
    print('> y = {} + {} * x + e <'.format(results.params[0],
                                           results.params[1]).center(80, '-'))
    
    title_print('Problem 6.b')
    print('> R^2 = {} | High R^2 value suggests statistical significant <'.\
          format(results.rsquared).center(80, '-'))
    
    title_print('Problem 6.c')
    fig, ax = plt.subplots()
    ax.scatter(results.fittedvalues, results.resid)
    ax.axhline(0)
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    plt.title('Residuals Versus Predicted Response')
    plt.show()
    print('> Erratic residuals when plotting vs. predicted response <'.\
          center(80, '-'))
    print('> Likely more complicated model would fit better <'.center(80, '-'))

def Problem_7():
    # TODO b, c, d, e, f, g, h1, h2, h3
    
    df = pd.DataFrame(data = {'y': # Miles/gal
                              [18.90, 17.00, 20.00, 18.25,
                               20.07, 11.20, 22.12, 21.47,
                               34.70, 30.40, 16.50, 36.50,
                               21.50, 19.70, 20.30, 17.80,
                               14.39, 14.89, 17.80, 16.41,
                               23.54, 21.47, 16.59, 31.90,
                               29.40, 13.27, 23.90, 19.73,
                               13.90, 13.27, 13.77, 16.50],
                              'X1': # Displacement (in^3)
                              [350, 350, 250, 351,
                               225, 440, 231, 262,
                               89.7, 96.9, 350, 85.3,
                               171, 258, 140, 302,
                               500, 440, 350, 318,
                               231, 360, 400, 96.9,
                               140, 460, 133.6, 318,
                               351, 351, 360, 350],
                              'X2': # Weight (lbs)
                              [3910, 3860, 3510, 3890,
                               3365, 4215, 3020, 3180,
                               1905, 2320, 3885, 2009,
                               2655, 3375, 2700, 3890,
                               5290, 5185, 3910, 3660,
                               3050, 4250, 3850, 2275,
                               2150, 5430, 2535, 4370,
                               4540, 4715, 4215, 3660]},
                      index = ['Apollo', 'Omega', 'Nova', 'Monarch',
                               'Duster', 'Jenson Conv.', 'Skyhawk', 'Monza',
                               'Scirocco', 'Corolla SR-5', 'Camaro',
                               'Datsun B210', 'Capri II', 'Pacer', 'Bobcat',
                               'Granada', 'Eldorado', 'Imperial', 'Nova LN',
                               'Valiant', 'Starfire', 'Cordoba', 'Trans Am',
                               'Corolla E-5', 'Astre', 'Mark IV', 'Celica GT',
                               'Charger SE', 'Cougar', 'Elite', 'Matador',
                               'Corvette'])
    y, X = patsy.dmatrices('y ~ X1 + X2', df)
    model = sm.OLS(y, X)
    results = model.fit()
    results.model.data.design_info = X.design_info
    
    title_print('Problem 7.a')
    print('y = {} + {} * x1 + {} * x2 + e'.format(round(results.params[0], 3),
                                                  round(results.params[1], 3),
                                                  round(results.params[2], 3))\
          .center(80, '-'))
    
    title_print('Problem 7.b')
    
    title_print('Problem 7.c')
    
    title_print('Problem 7.d')
    
    title_print('Problem 7.e')
    
    title_print('Problem 7.f')
    
    title_print('Problem 7.g')
    
    #################
    # Problem 7.h.1 #
    #################
    title_print('Problem 7.h.1')
    residuals = results.resid
    prob = [(i - 1/2) / len(y) for i in range(len(y))]
    
    # Can plot straight line for visuals
    resid_results = sm.OLS(prob, sm.add_constant(sorted(residuals))).fit()
    X_range = np.linspace(min(residuals), max(residuals), len(residuals))
    
    # Normal Probability Plot + straight line
    fig, ax = plt.subplots()
    ax.scatter(sorted(residuals), prob)
    ax.plot(X_range,
            resid_results.params[0] + resid_results.params[1] * X_range)
    ax.set_xlabel('Residual')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    plt.title('Normal Probability Plot')
    plt.show()
    
    print('> Does not appear to be problem with normality <'.center(80, '-'))

    #################
    # Problem 7.h.2 #
    #################
    title_print('Problem 7.h.2')
    fig, ax = plt.subplots()
    ax.scatter(results.fittedvalues, residuals)
    ax.axhline(0)
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    plt.title('Residuals Versus Predicted Response')
    plt.show()
    
    print('> Definite non-linear pattern. Either slight downward trend <'.\
          center(80, '-'))
    print('> if you disregard 5 points in upper right. OR somewhat <'.\
          center(80, '-'))
    print('> quadratic if disregard 3 points in lower right <'.center(80, '-'))
    
    #################
    # Problem 7.h.3 #
    #################
    title_print('Problem 7.h.3')
    fig, ax = plt.subplots()
    ax2 = ax.twiny()
    scat_1 = ax.scatter(df['X1'], residuals, color = 'orange', label = 'X1')
    scat_2 = ax2.scatter(df['X2'], residuals, color = 'black', label = 'X2')
    ax.axhline(0)
    ax.set_xlabel('X_1')
    ax2.set_xlabel('X_2')
    ax.set_ylabel('Residuals')
    
    plots = scat_1 + scat_2
    labels = [label.get_label() for label in plots]
    ax.legend(plots, labels)
    ax2.legend()
    
    plt.title('Residuals Versus X_i')
    plt.show()

    return df, results


