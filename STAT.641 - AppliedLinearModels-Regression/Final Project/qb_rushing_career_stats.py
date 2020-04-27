#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:23:55 2019

@author: Jake
"""

#############################################################################
# H_0: Rushing has no impact on length of QB careers                        #
# H_a: QBs with higher than average rushing stats have shorter than average #
#      careers                                                              #
#############################################################################

import math
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import patsy
from scipy.stats import f
from itertools import compress

def title_print(text):
    '''
    Used throughout to print section titles
    '''
    print()
    print('#' * (len(text) + 4))
    print('|', text, '|')
    print('#' * (len(text) + 4))

# Load data from CSV
df = pd.read_csv('qb_rushing.csv')

# Organize by player rank
df = df.set_index('player_rank').sort_index()
print(df.head(10).to_string())

# Total QBs
print('Total QBs: {}'.format(df.shape[0]))

# Look at players with >1 year, started >= 10 games, since first Super Bowl
df = df[(df['first_year'] < df['last_year']) &
        (df['games_started'] >= 10) &
        (df['first_year'] > 1966)]
print('QBs who played >1 year, started 10+ games, since 1966-67 season: {}'\
      .format(df.shape[0]))

# Use these 4 categories for analysis - note the NaN values in YPA
categories = ['games_played', 'games_started', 'rushing_attempts',
              'yards_per_attempt', 'total_yards']
print()
print(df[categories].describe())
quantiles = [0.25, 0.5, 0.75]
gp_quant = np.quantile(df['games_played'], quantiles)
ra_quant = np.quantile(df['rushing_attempts'], quantiles)
ypa_quant = np.nanquantile(df['yards_per_attempt'], quantiles)

################
# GAMES PLAYED #
################
games_max = int(max(df['games_played']))
games_mean = round(df['games_played'].mean(), 1)
games_median = df['games_played'].median()

# Histogram of games played
plt.figure(figsize = (8, 8))
plt.hist(df['games_played'], [i for i in range(0, games_max, 10)])
plt.axvline(games_median,
            color = 'red',
            label = 'Median [{}]'.format(games_median))
plt.axvline(games_mean,
            color = 'black',
            label = 'Mean [{}]'.format(games_mean))
plt.title('Games Played per QB (1 bin = 10 games)')
plt.xlabel('Games Played')
plt.ylabel('Count')
plt.xlim(0, 10 * math.floor(games_max/10))
plt.grid(which = 'major')
plt.legend(loc = 'best')
plt.show()

####################
# RUSHING ATTEMPTS #
####################
attempts_max = int(max(df['rushing_attempts']))
attempts_mean = round(df['rushing_attempts'].mean(), 1)
attempts_median = round(df['rushing_attempts'].median(), 1)

# Histogram of rushing attempts
plt.figure(figsize = (8, 8))
plt.hist(df['rushing_attempts'], [i for i in range(0, attempts_max, 25)])
plt.axvline(attempts_median,
            color = 'red',
            label = 'Median [{}]'.format(attempts_median))
plt.axvline(attempts_mean,
            color = 'black',
            label = 'Mean [{}]'.format(attempts_mean))
plt.xlim(0, 1200)
plt.title('Rushing Attempts (1 bin = 25 attempts)')
plt.xlabel('Attempts')
plt.ylabel('Count')
plt.grid(which = 'major')
plt.legend(loc = 'best')
plt.show()

###############################
# YARDS PER ATTEMPT - ALL QBS #
###############################
yards_max = int(max(df['yards_per_attempt']))
yards_mean = round(df['yards_per_attempt'].mean(), 1)
yards_median = round(df['yards_per_attempt'].median(), 1)

# Histogram of yards per attempt
plt.figure(figsize = (8, 8))
plt.hist(df['yards_per_attempt'], bins = [i for i in range(-2, 11)])
plt.axvline(yards_median,
            color = 'red',
            label = 'Median [{}]'.format(yards_median))
plt.axvline(yards_mean,
            color = 'black',
            label = 'Mean [{}]'.format(yards_mean))
plt.xlim(-2, 10)
plt.title('Rushing Yards per Attempt (1 bin = 1 yard)')
plt.xlabel('Yards per Attempt')
plt.ylabel('Count')
plt.grid(which = 'major')
plt.legend(loc = 'best')
plt.show()

################################################################
# COMPARE LENGTH OF CAREER OF TOP 25% TO BOTTOM 25% - ATTEMPTS #
################################################################
# This doesn't control for bad QBs, so look at yards per attempt
print()
att_top25 = df[df['rushing_attempts'] > ra_quant[2]]
att_bot25 = df[df['rushing_attempts'] < ra_quant[0]]
att_top25_mean_games = round(att_top25['games_played'].mean(), 1)
att_top25_median_games = round(att_top25['games_played'].median(), 1)
att_bot25_mean_games = round(att_bot25['games_played'].mean(), 1)
att_bot25_median_games = round(att_bot25['games_played'].median(), 1)
print('Mean games played of top 25% of QBs [count: {}] based on rushing '
      'attempts: {}'.format(att_top25.shape[0], att_top25_mean_games))
print('Mean games played of bottom 25% of QBs [count: {}] based on rushing '
      'attempts: {}'.format(att_bot25.shape[0], att_bot25_mean_games))

#########################################################################
# COMPARE LENGTH OF CAREER OF TOP 25% TO BOTTOM 25% - YARDS PER ATTEMPT #
#########################################################################
print()
ypa_top25 = df[df['yards_per_attempt'] > ypa_quant[2]]
ypa_bot25 = df[df['yards_per_attempt'] < ypa_quant[0]]
ypa_top25_mean_games = round(ypa_top25['games_played'].mean(), 1)
ypa_top25_median_games = round(ypa_top25['games_played'].median(), 1)
ypa_bot25_mean_games = round(ypa_bot25['games_played'].mean(), 1)
ypa_bot25_median_games = round(ypa_bot25['games_played'].median(), 1)
print('Mean games played, top 25% of QBs [count: {}] based on yards per '
      'attempt: \n---> {} <---'.format(ypa_top25.shape[0],
                                       ypa_top25_mean_games))
print('Mean games played, bottom 25% of QBs [count: {}] based on yards per '
      'attempt: \n---> {} <---'.format(ypa_bot25.shape[0],
                                       ypa_bot25_mean_games))

########################################################################
# HISTOGRAM OF GAMES PLAYED FOR TOP AND BOTTOM 25% - YARDS PER ATTEMPT #
########################################################################
print()
gp_top25_max = int(ypa_top25['games_played'].max())
gp_bot25_max = int(ypa_bot25['games_played'].max())
plt.hist(ypa_top25['games_played'], [i for i in range(0, gp_top25_max, 25)],
         alpha = 0.75, label = 'Top 25%')
plt.hist(ypa_bot25['games_played'], [i for i in range(0, gp_bot25_max, 25)],
         alpha = 0.5, label = 'Bot 25%')
plt.legend()
plt.show()

##############################################################
# STANDARD DEVIATIONS OF GAMES PLAYED FOR TOP AND BOTTOM 25% #
##############################################################
print('Standard deviation of games played for top 25%: {}'.
      format(round(np.std(ypa_top25['games_played']), 2)))
print('Standard deviation of games played for bottom 25%: {}'.
      format(round(np.std(ypa_bot25['games_played']), 2)))

##########################
# PAIRPLOT OF CATEGORIES #
##########################
plt.figure(figsize = (8, 8))
fig = sns.pairplot(df[categories])
plt.show()

#############################
# PLOT ATTEMPTS/YARDS/GAMES #
#############################
fig, ax = plt.subplots(figsize = (8, 8))
ax.scatter(df['rushing_attempts'],
           df['yards_per_attempt'],
           s = df['games_played'],
           c = df['games_played'],
           cmap = mpl.cm.cool,
           alpha = 0.7)
ax.grid()
ax.set_xlabel('Rushing Attempts')
ax.set_ylabel('Yards Per Attempt')
ax.set_title('Games played as function of rushing attempts & ' \
             'yards per attempt')
plt.show()

class Analysis:
    '''
    Class to run ordinary least squares (OLS) and associ
    '''
def run_analysis(df, drop_point = None):
    '''
    Parameters
    ----------
    drop_point : int or list-like, optional
        Point(s) to drop for analysis. The default is None.

    Returns
    -------
    Points to potentially drop, if drop_point == None.

    '''
    #####################################
    # DROP INFLUENTIAL POINTS IF NEEDED #
    #####################################
    if drop_point:
        df = df.reset_index().drop(drop_pts).set_index('player_rank')
    y, X = patsy.dmatrices('games_played ~ rushing_attempts + total_yards',
                           df)

    ###########################
    # LINEAR REGRESSION MODEL #
    ##########################
    model = sm.OLS(y, X)
    results = model.fit()
    results.model.data.design_info = X.design_info
    coefs = np.round(results.params, 3)
    print()
    print(results.summary())
    title_print('Model')
    print('y = {} + {} * rushing_attempts + {} * total_yards'.\
          format(coefs[0], coefs[1], coefs[2]))
    
    ###########################
    # DROP INFLUENTIAL POINTS #
    ###########################
    # If dropping points, only run to here and then exit function
    title_print('Significance')
    if drop_point:
        print('Points dropped: {}'.format(drop_point))
        print('Coefficients: {}'.format(np.round(results.params, 3)))
        print('R-squared: {}'.format(round(results.rsquared, 3)))
        return
    else:
        print('Coefficients: {}'.format(np.round(results.params, 3)))
        print('R-squared: {}'.format(round(results.rsquared, 3)))
    
    ##############################
    # ANOVA TABLE / SIGNIFICANCE #
    ##############################
    # H0: beta_0 = beta_1 = beta_2
    # H1: beta_j != 0
    # Rushing attempts more significant that total yards
    # Makes sense because more time in league closely associated with
    # more chances to run
    aov_table = sm.stats.anova_lm(results, typ = 1)
    title_print('Analysis of Variance table')
    print(aov_table)
    print('\nCalculated F-stat: {}'.
          format(round(f.ppf(0.025, X.shape[1] - 1, X.shape[0]), 3)))
    print('Regression F: {}'.format(round(results.fvalue, 2)))
    print('Regression p: {}'.format(round(results.f_pvalue, 4)))
    print('---> Regression is significant <---')
    
    ########################
    # CONFIDENCE INTERVALS #
    ########################
    conf_int = np.round(results.conf_int(), 3)
    print()
    title_print('95% Confidence Intervals')
    print('Intercept: {} to {}'.format(conf_int[0][0], conf_int[0][1]))
    print('Rushing Attempts: {} to {}'.format(conf_int[1][0], conf_int[1][1]))
    print('Total yards: {} to {}'.format(conf_int[2][0], conf_int[2][1]))

    #####################
    # MULTICOLLINEARITY #
    #####################
    vif = np.round([variance_inflation_factor(X, i)
                    for i in range(X.shape[1])], 4)
    title_print('Multicollinearity')
    [print('VIF_{}: {}'.format(i, vif[i])) for i, v in enumerate(vif)]

    #############
    # RESIDUALS #
    #############
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
    
    # Calculate OLS using resid to plot straight line. Get y values from model
    resid_results = sm.OLS(Prob, sm.add_constant(sorted(residuals))).fit()
    X_range = np.linspace(min(residuals), max(residuals), len(residuals))
    
    # Normality plot
    fig = plt.figure(figsize = (8, 8))
    plt.scatter(sorted(residuals), Prob)
    plt.plot(X_range,
             resid_results.params[0] + resid_results.params[1] * X_range)
    plt.xlabel('Residual')
    plt.ylabel('Probability')
    plt.title('Normal Probability Plot')
    plt.show()
    print('---> Heavy-tailed distribution <---')
    
    ############
    # OUTLIERS #
    ############
    title_print('Outliers / Influence Points')
    pos_out = (np.argmax(residuals), np.amax(residuals))
    neg_out = (np.argmax(-residuals), -np.amax(-residuals))
    x_out = (np.argmax(results.fittedvalues), np.amax(results.fittedvalues))
    # Visually from residual plot, these 3 points are outliers
    
    # Influential points
    infl = results.get_influence()
    infl_df = infl.summary_frame()
    print(infl_df.head().to_string())
    print('...continued...')
    infl_pts = {}
    
    # Leverage Points - Hat Diagonal
    n, p = X.shape[0], X.shape[1] - 1
    lev_pt = 2 * p / n
    dhat_pts = list(infl_df[infl_df['hat_diag'] > lev_pt].index)
    print('\n***| Hat Diagonal |***')
    print('Leverage calculation (2 * p \ n) = {}'.format(round(lev_pt, 3)))
    print('Points where hat diagonal exceeds leverage calculation: {}'.
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
    print('Points which exceed DFFITS cutoff: {}'.
          format(DFFITS_pts))
    
    # DFBETAS
    print('\n***| DFBETAS |***')
    DFBETAS_cutoff = 2 / np.sqrt(n)
    DFBETAS_pts = []
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
    print('Points which are greater than COVRATIO upper bound cutoff: {}'.
          format(gt_cutoff))
    print('Points which are less than COVRATIO lower bound cutoff: {}'.
          format(lt_cutoff))
    
    # Most influential points
    for i in dhat_pts + cook_pts + DFFITS_pts + DFBETAS_pts + COVRATIO_pts:
        infl_pts[i] = infl_pts.get(i, 0) + 1
    most_infl = [pt for pt in infl_pts
                 if infl_pts[pt] == max(infl_pts.values())]
    print('\n***| MOST INFLUENTIAL POINTS |***') #points in every cutoff
    print(sorted(most_infl))
    
    # Check who these points are
    return X, y, most_infl

X, y, drop_pts = run_analysis(df, drop_point = None)
run_analysis(df, drop_pts)