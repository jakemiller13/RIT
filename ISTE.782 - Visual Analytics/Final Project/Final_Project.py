#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 11:09:11 2021

@author: jmiller
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from functools import reduce
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

hospitals = {'0': [0.180960, -119.959400],
             '1': [0.153120, -119.915900],
             '2': [0.151090, -119.909520],
             '3': [0.121800, -119.904300],
             '4': [0.134560, -119.883420],
             '5': [0.182990, -119.855580],
             '6': [0.041470, -119.828610],
             '7': [0.065250, -119.744800]}
diff = 0.01

df = pd.read_csv('~/Google Drive/RIT/ISTE.782 - Visual Analytics/'
                 + 'Final Project/MC1/mc1-reports-data.csv')
df2 = pd.read_csv('~/Google Drive/RIT/ISTE.782 - Visual Analytics/'
                  + 'Final Project/MobileSensorReadings.csv')
# =============================================================================
# Plot 1
# =============================================================================
df['time'] = pd.to_datetime(df['time'])

fig1, ax1 = plt.subplots()
plot_1 = df.plot(x = 'time',
                 y = 'location',
                 s = 'shake_intensity',
                 kind = 'scatter',
                 yticks = range(1, 20),
                 title = 'Shake Intensity Per Location',
                 ax = ax1,
                 rot = 45,
                 alpha = 0.2)
ax1.set_xlabel('Date')
ax1.set_ylabel('Location')
ax1.grid()
plt.show()

# =============================================================================
# Plot 2
# =============================================================================
df1 = df.copy()
df1['time'] = df1['time'].round('60min')

df.plot(x = 'time',
        y = 'location',
        s = 'shake_intensity',
        kind = 'scatter',
        yticks = range(1, 20))
plt.show()

# =============================================================================
# Plot 3
# =============================================================================
adj_color = cm.get_cmap('tab20', 19)

fig3, ax3 = plt.subplots()
plot_3 = df1.plot(x = 'time',
                  # y = 'medical',
                  y = 'location',
                  s = 'shake_intensity',
                  # c = 'location',
                  # c = 'medical',
                  # cmap = adj_color,
                  yticks=range(1, 20),
                  kind = 'scatter',
                  title = 'Shake Intensity for Medical Buildings at all Locations',
                  ax = ax3,
                  rot = 45,
                  alpha = 0.2)
ax3.set_xlabel('Date')
# ax3.set_ylabel('Medical')
ax3.set_ylabel('Location')
ax3.grid(axis = 'x')
plt.tight_layout()
# plt.savefig('/Users/jmiller/Desktop/Med_Shake_Intensity.png')

# =============================================================================
# Set up hospital areas
# =============================================================================

radiation = {}
for i in hospitals:
    h = hospitals[i]
    rad = []
    lat = [h[0] - diff, h[0] + diff]
    lon = [h[1] - diff, h[1] + diff]
    rad = df2[(df2['Long'].between(lon[0], lon[1]) &
               df2['Lat'].between(lat[0], lat[1]))]
    rad = rad[rad['Value'] > 100]  # Arbitrary basline?
    rad = rad.rename(columns = {'Value': i})

    radiation[i] = rad[['Timestamp', i]]

df_r = [pd.DataFrame(radiation[i]) for i in radiation]
df_final = reduce(lambda left, right: pd.merge(left, right,
                                               on = ['Timestamp'],
                                               how = 'outer'), df_r)
df_final['Timestamp'] = pd.to_datetime(df_final['Timestamp'])

colors = ['red', 'orange', 'yellow', 'green',
          'aqua', 'blue', 'magenta', 'violet']

for h in df_final.columns[1:]:
    fig4, ax4 = plt.subplots()
    tmp_df = df_final[['Timestamp', h]]
    ax4.plot(tmp_df['Timestamp'], tmp_df[h], color = colors[int(h)])
    for i in df_final.columns[1:]:
        tmp_df = df_final[['Timestamp', i]]
        ax4.scatter(x = tmp_df['Timestamp'],
                    y = tmp_df[i],
                    color = colors[int(i)],
                    label = 'Hospital {}'.format(i),
                    linestyle = 'solid',
                    marker = '.')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Radiation CPM')
    ax4.set_ylim([0, 1500])
    # ax4.set_xticklabels(labels = list(tmp_df['Timestamp']), rotation = 45)
    # ax4.legend(loc = 'lower left')
    ax4.grid(axis = 'x')
    ax4.set_title('Radiation CPM per Hospital (Hospital {})'.format(h))
    fig4.show()
    fig4.savefig('/Users/jmiller/Desktop/Rad_Hosp_{}.png'.format(h))
