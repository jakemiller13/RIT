#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 08:49:27 2019

@author: Jake
"""

import matplotlib.pyplot as plt
import pandas as pd

# Dataframes
edu = pd.read_csv('Combined_Education.tsv',
                  delimiter = '\t',
                  index_col = 'country')
inc = pd.read_csv('Combined_Income.tsv',
                  delimiter = '\t',
                  index_col = 'Country Name')
mor = pd.read_csv('Combined_Mortality.tsv',
                  delimiter = '\t',
                  index_col = 'Country Name')
ter = pd.read_csv('Combined_Terrorism.tsv',
                  delimiter = '\t',
                  index_col = 'Country')
# To align dataframes
ter.columns = ter.columns.str.replace(' ', '')
ter.drop(columns = '2017', inplace = True)

######################################
# PLOT THREE TOGETHER FOR COMPARISON #
######################################

# Create figure for 3 plots
fig = plt.figure(figsize = (10, 10))
fig.suptitle('Business Expansion into Israel, Japan or Mexico',
             y = 1.02,
             fontsize = 15)
gs = fig.add_gridspec(2, 2)

# Income Plot
del inc.index.name
ax1 = fig.add_subplot(gs[0, :])
inc.T.plot(kind = 'line', ax = ax1)
ax1.set_title('Net National Income', fontsize = 13)
ax1.set_xlim(0, inc.shape[1]-1)
ax1.set_ylim(0, 40000)
ax1.set_xlabel('Year')
ax1.set_ylabel('Income per Capita')
ax1.grid(which = 'major', axis = 'both')
plt.savefig('Plot_Income.png')

# Mortality Plot
del mor.index.name
ax2 = fig.add_subplot(gs[1, 0])
mor.T.plot(kind = 'line', ax = ax2)
ax2.set_title('Infant Mortality', fontsize = 13)
ax2.set_xlim(0, inc.shape[1]-1)
ax2.set_ylim(0, 25)
ax2.set_xlabel('Year')
ax2.set_ylabel('Deaths per 1000 Live Births')
ax2.grid(which = 'major', axis = 'both')

# Terrorism Plot
del ter.index.name
ax3 = fig.add_subplot(gs[1, 1])
ter.T.plot(kind = 'line', ax = ax3)
ax3.set_title('Terrorist Acts', fontsize = 13)
ax3.set_xlim(0, inc.shape[1]-1)
ax3.set_ylim(0, 150)
ax3.set_xlabel('Year')
ax3.set_ylabel('Successful Acts')
ax3.grid(which = 'major', axis = 'both')

plt.savefig('Plot_Income_Mortality_Terrorism.png')
plt.tight_layout()

# Education Plot, separate because of difference in time frames
fig2, ax = plt.subplots(figsize = (5, 5))
del edu.index.name
edu.T.plot(kind = 'line', ax = ax)
ax.set_title('Educational Expenditure', fontsize = 13)
ax.set_xlim(0, edu.shape[1]-1)
ax.set_ylim(0, 8)
ax.set_xlabel('Year')
ax.set_ylabel('Percent of GDP')
ax.grid(which = 'major', axis = 'both')
plt.savefig('Plot_Education.png')
plt.show()

#######################################
# PLOT EACH INDIVIDUALLY FOR WRITE-UP #
#######################################

# Income Plot
inc.T.plot(kind = 'line', figsize = (5, 5))
plt.title('Net National Income', fontsize = 13)
plt.xlim(0, inc.shape[1]-1)
plt.ylim(0, 40000)
plt.xlabel('Year')
plt.ylabel('Income per Capita')
plt.grid(which = 'major', axis = 'both')
plt.tight_layout()
plt.savefig('Plot_Income.png')

# Mortality Plot
mor.T.plot(kind = 'line', figsize = (5, 5))
plt.title('Infant Mortality', fontsize = 13)
plt.xlim(0, inc.shape[1]-1)
plt.ylim(0, 25)
plt.xlabel('Year')
plt.ylabel('Deaths per 1000 Live Births')
plt.grid(which = 'major', axis = 'both')
plt.tight_layout()
plt.savefig('Plot_Mortality.png')

# Terrorism Plot
ter.T.plot(kind = 'line', figsize = (5, 5))
plt.title('Terrorist Acts', fontsize = 13)
plt.xlim(0, inc.shape[1]-1)
plt.ylim(0, 150)
plt.xlabel('Year')
plt.ylabel('Successful Acts')
plt.grid(which = 'major', axis = 'both')
plt.tight_layout()
plt.savefig('Plot_Terrorism.png')