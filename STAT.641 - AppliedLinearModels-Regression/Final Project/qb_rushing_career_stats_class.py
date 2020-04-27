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
# df = df.set_index('player_rank').sort_index()
df = df.sort_values('player_rank').reset_index(drop = True)
print(df.head(10).to_string())

# Total QBs
print('Total QBs: {}'.format(df.shape[0]))

# Look at players with >1 year, started >= 10 games, since first Super Bowl
df = df[(df['first_year'] < df['last_year']) &
        (df['games_started'] >= 10) &
        (df['first_year'] >= 1966)]
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

# Distplot of games played
plt.figure(figsize = (8, 8))
sns.distplot(df['games_played'], [i for i in range(0, games_max, 10)])
plt.axvline(games_median,
            color = 'red',
            label = 'Median [{}]'.format(games_median))
plt.axvline(games_mean,
            color = 'black',
            label = 'Mean [{}]'.format(games_mean))
plt.title('Games Played per QB (1 bin = 10 games)')
plt.xlabel('Games Played')
plt.ylabel('Density')
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

# Distplot of rushing attempts
plt.figure(figsize = (8, 8))
sns.distplot(df['rushing_attempts'], [i for i in range(0, attempts_max, 25)])
plt.axvline(attempts_median,
            color = 'red',
            label = 'Median [{}]'.format(attempts_median))
plt.axvline(attempts_mean,
            color = 'black',
            label = 'Mean [{}]'.format(attempts_mean))
plt.xlim(0, 1200)
plt.title('Rushing Attempts (1 bin = 25 attempts)')
plt.xlabel('Attempts')
plt.ylabel('Density')
plt.grid(which = 'major')
plt.legend(loc = 'best')
plt.show()

###############################
# YARDS PER ATTEMPT - ALL QBS #
###############################
yards_max = int(max(df['yards_per_attempt']))
yards_mean = round(df['yards_per_attempt'].mean(), 1)
yards_median = round(df['yards_per_attempt'].median(), 1)

# Distplot of yards per attempt
plt.figure(figsize = (8, 8))
sns.distplot(df['yards_per_attempt'], bins = [i for i in range(-2, 11)])
plt.axvline(yards_median,
            color = 'red',
            label = 'Median [{}]'.format(yards_median))
plt.axvline(yards_mean,
            color = 'black',
            label = 'Mean [{}]'.format(yards_mean))
plt.xlim(-2, 10)
plt.title('Rushing Yards per Attempt (1 bin = 1 yard)')
plt.xlabel('Yards per Attempt')
plt.ylabel('Density')
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

##############################################################
# STANDARD DEVIATIONS OF GAMES PLAYED FOR TOP AND BOTTOM 25% #
##############################################################
print('Standard deviation of games played for top 25%: {}'.
      format(round(np.std(ypa_top25['games_played']), 2)))
print('Standard deviation of games played for bottom 25%: {}'.
      format(round(np.std(ypa_bot25['games_played']), 2)))

########################################################################
# HISTOGRAM OF GAMES PLAYED FOR TOP AND BOTTOM 25% - YARDS PER ATTEMPT #
########################################################################
print()
plt.figure(figsize = (8, 8))
gp_top25_max = int(ypa_top25['games_played'].max())
gp_bot25_max = int(ypa_bot25['games_played'].max())
sns.distplot(ypa_top25['games_played'],
             [i for i in range(0, gp_top25_max, 10)],
             label = 'Top 25%')
sns.distplot(ypa_bot25['games_played'],
             [i for i in range(0, gp_bot25_max, 10)],
             label = 'Bot 25%')
plt.xlim(0, 300)
plt.xlabel('Games Played')
plt.ylabel('Density')
plt.title('Career Games Played for Top and Bottom 25% of Rushing QBs' + \
          '\n(1 bin = 10 games)')
plt.grid(which = 'major')
plt.legend()
plt.show()

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
           df['total_yards'],
           s = df['games_played'],
           c = df['games_played'],
           cmap = mpl.cm.cool,
           alpha = 0.7)
ax.grid()
ax.set_xlabel('Rushing Attempts')
ax.set_ylabel('Total Yards')
ax.set_title('Games played as function of rushing attempts & ' \
             'yards per attempt')
plt.show()

class Analysis:
    '''
    Class to run ordinary least squares (OLS) and associated analysis
    '''
    
    def __init__(self, df_in, drop_pts = None):
        '''
        Parameters
        ----------
        df_in : dataframe to perform OLS on
        drop_points : int or list-like, optional
            Point(s) to drop for analysis. The default is None.
    
        Returns
        -------
        Points to potentially drop, if drop_points == None.
    
        '''        
        self.drop_points = drop_pts
        self.df = df_in.reset_index(drop = True)
        if self.drop_points:
            self.df = self.df.drop(self.drop_points).reset_index(drop = True)

        self.y, self.X = patsy.dmatrices('games_played ~ \
                                         rushing_attempts + total_yards',
                                         self.df)

    def lin_mod(self):
        '''
        Returns
        -------
        results : results class - use "dir(results)" to see available data

        '''
        self.model = sm.OLS(self.y, self.X)
        self.results = self.model.fit()
        self.results.model.data.design_info = self.X.design_info
        self.coefs = np.round(self.results.params, 3)
        print(self.results.summary())
        title_print('Model')
        print('y = {} + {} * rushing_attempts + {} * total_yards'.\
              format(self.coefs[0], self.coefs[1], self.coefs[2]))
        return self.results
    
    def significance(self):
        '''
        Returns
        -------
        None. Print R-squared

        '''
        title_print('Significance')
        if self.drop_points:
            print('Points dropped: {}'.format(sorted(self.drop_points)))
            print('Coefficients: {}'.format(np.round(self.results.params, 3)))
            print('R-squared: {}'.format(round(self.results.rsquared, 3)))
            return
        else:
            print('Coefficients: {}'.format(np.round(self.results.params, 3)))
            print('R-squared: {}'.format(round(self.results.rsquared, 3)))
    
    def anova(self):
        '''         
        Returns
        -------
        None. Prints ANOVA table

        '''
        # H0: beta_0 = beta_1 = beta_2
        # H1: beta_j != 0
        # Rushing attempts more significant that total yards
        # Makes sense because more time in league closely associated with
        # more chances to run
        aov_table = sm.stats.anova_lm(self.results, typ = 1)
        title_print('Analysis of Variance table')
        print(aov_table)
        print('\nCalculated F-stat: {}'.
              format(round(f.ppf(0.025, self.X.shape[1] - 1, self.X.shape[0]),
                           3)))
        print('Regression F: {}'.format(round(self.results.fvalue, 2)))
        print('Regression p: {}'.format(round(self.results.f_pvalue, 4)))
        print('---> Regression is significant <---')
    
    def confidence_interval(self, a = 0.05):
        '''
        Parameters
        ----------
        a: float, alpha for confidence interval. Default: 0.05
            Note: use alpha of 0.05 for 95% confidence interval, for example
        
        Returns
        -------
        None. Calculates and prints confidence intervals

        '''
        conf_int = np.round(self.results.conf_int(a), 3)
        title_print('95% Confidence Intervals')
        print('Intercept: {} to {}'.format(conf_int[0][0], conf_int[0][1]))
        print('Rushing Attempts: {} to {}'.format(conf_int[1][0],
                                                  conf_int[1][1]))
        print('Total yards: {} to {}'.format(conf_int[2][0], conf_int[2][1]))

    def multicollinearity(self):
        '''
        Returns
        -------
        None. Calculates and prints variance inflation factor for parameters

        '''
        vif = np.round([variance_inflation_factor(self.X, i)
                        for i in range(self.X.shape[1])], 4)
        title_print('Multicollinearity')
        [print('VIF_{}: {}'.format(i, vif[i])) for i, v in enumerate(vif)]

    def residuals(self):
        '''
        Returns
        -------
        None. Calculates residuals. Plots residuals vs. fitted values and
              normality plots.

        '''
        self.resid = self.results.resid
        Prob = [(i - 1/2) / len(self.y) for i in range(len(self.y))]
        
        # Plot residuals vs. fitted values
        fig, ax = plt.subplots(figsize = (8, 8))
        ax.scatter(self.results.fittedvalues, self.resid)
        ax.axhline(0)
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('Residuals')
        plt.title('Residuals Versus Predicted Response')
        plt.show()
        
        # Calculate OLS from resid to plot straight line. y values from model
        resid_results = sm.OLS(Prob, sm.add_constant(sorted(self.resid))).fit()
        X_range = np.linspace(min(self.resid),
                              max(self.resid),
                              len(self.resid))
        
        # Normality plot
        fig = plt.figure(figsize = (8, 8))
        plt.scatter(sorted(self.resid), Prob)
        plt.plot(X_range,
                 resid_results.params[0] + resid_results.params[1] * X_range)
        plt.xlabel('Residual')
        plt.ylabel('Probability')
        plt.title('Normal Probability Plot')
        plt.show()
        print('---> Heavy-tailed distribution <---')
    
    def outliers(self):
        '''
        Find outliers and influential points based on:
              -leverage points (hat diagonal)
              -Cook's D
              -DFFITS
              -DFBETAS
              -COVRATIO
        Also prints points that appear in all tests
        
        Returns
        -------
        list : Points that appear in all tests
        
        '''
        title_print('Outliers / Influence Points')
        pos_out = (np.argmax(self.resid), np.amax(self.resid))
        neg_out = (np.argmax(-self.resid), -np.amax(-self.resid))
        x_out = (np.argmax(self.results.fittedvalues),
                 np.amax(self.results.fittedvalues))
        # Visually from residual plot, these 3 points are outliers
        
        # Influential points
        infl = self.results.get_influence()
        infl_df = infl.summary_frame()
        print(infl_df.head())
        print('...continued...')
        infl_pts = {}
        
        # Leverage Points - Hat Diagonal
        n, p = self.X.shape[0], self.X.shape[1] - 1
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
        
        # Check who these points are
        return most_infl

# First run through analysis
title_print('Run 1')
run_1 = Analysis(df)
results_1 = run_1.lin_mod()
run_1.significance()
run_1.anova()
run_1.confidence_interval()
run_1.multicollinearity()
run_1.residuals()
drop_pts_1 = run_1.outliers()

# Print and plot outliers
outlier_df = pd.DataFrame(columns = ['player',
                                     'games_played',
                                     'rushing_attempts',
                                     'total_yards',
                                     'yards_per_attempt'])
for i in drop_pts_1:
    player_info = df.iloc[i][['player',
                              'games_played',
                              'rushing_attempts',
                              'total_yards',
                              'yards_per_attempt']]
    outlier_df = outlier_df.append(player_info)
    print(player_info)
    print()

fig, ax = plt.subplots(figsize = (8, 8))
ax.scatter(outlier_df['rushing_attempts'],
           outlier_df['total_yards'],
           s = outlier_df['games_played'],
           c = outlier_df['games_played'],
           cmap = mpl.cm.cool,
           alpha = 0.7)
ax.grid()
ax.set_xlabel('Rushing Attempts')
ax.set_ylabel('Total Yards')
ax.set_title('Games played as function of rushing attempts & ' \
             'yards per attempt - Outliers')
plt.show()

# Run 2 with outliers dropped
title_print('Run 2')
run_2 = Analysis(df, drop_pts_1)
results_2 = run_2.lin_mod()
run_2.significance()
run_2.anova()
run_2.confidence_interval()
run_2.multicollinearity()
run_2.residuals()
drop_pts_2 = run_2.outliers()

# Print outliers
df_2 = df.drop(drop_pts_2).reset_index(drop = True)
for i in drop_pts_2:
    print(df_2.iloc[i][['player', 'games_played', 'rushing_attempts',
                      'total_yards', 'yards_per_attempt']])
    print()