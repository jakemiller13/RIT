#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 11:56:50 2021

@author: jmiller
"""

# Package imports
import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
opts.defaults(opts.HeatMap(radial = True, width = 800, height = 800,
                           tools = ["hover"]))

# Data import
df_hcin = pd.read_csv('~/Google Drive/RIT/ISTE.782 - Visual Analytics/Week 6/data/HCIN.csv')
df_iste = pd.read_csv('~/Google Drive/RIT/ISTE.782 - Visual Analytics/Week 6/data/ISTE.csv')
df_nssa = pd.read_csv('~/Google Drive/RIT/ISTE.782 - Visual Analytics/Week 6/data/NSSA.csv')

df_schools = [df_hcin, df_iste, df_nssa]

# Class room names
rooms = ['Golisano Hall (GOL)-2160',
         'Golisano Hall (GOL)-2320',
         'Golisano Hall (GOL)-2520',
         'Golisano Hall (GOL)-2620',
         'Golisano Hall (GOL)-2650',
         'Golisano Hall (GOL)-3510',
         'Golisano Hall (GOL)-3690']

# Preprate room dataframes
df_2160 = pd.DataFrame(columns = df_hcin.columns)
df_2320 = pd.DataFrame(columns = df_hcin.columns)
df_2520 = pd.DataFrame(columns = df_hcin.columns)
df_2620 = pd.DataFrame(columns = df_hcin.columns)
df_2650 = pd.DataFrame(columns = df_hcin.columns)
df_3510 = pd.DataFrame(columns = df_hcin.columns)
df_3690 = pd.DataFrame(columns = df_hcin.columns)

room_dfs = [df_2160, df_2320, df_2520, df_2620, df_2650, df_3510, df_3690]
room_dict = {rooms[i]: room_dfs[i] for i in range(len(rooms))}

# Separate rooms
for i in room_dict:
    for j in df_schools:
        room_dict[i] = pd.concat([room_dict[i], j[j['Room'] == i]])

# Setup dataframes
for i in room_dict:
    
    # Calculate percent filled
    room_dict[i]['Perc_Filled'] = \
        room_dict[i]['Enrollment'] / room_dict[i]['Capacity']
    
    # Split dates and times
    room_dict[i] = pd.concat([room_dict[i],
                              room_dict[i]['Days & Time'].\
                                  str.split(expand = True)], axis = 1)
    room_dict[i] = room_dict[i].rename(columns = {0: 'Days',
                              1: 'Start',
                              2: 'drop',
                              3: 'End'})
    
    # Split days
    days = room_dict[i]['Days'].replace({r"([A-Z][a-z])": r" \1"},
                                        regex = True).\
        str.split(expand = True)
    room_dict[i] = pd.concat([room_dict[i], days], axis = 1)
    room_dict[i] = room_dict[i].rename(columns = {0: 'Day_1',
                                                  1: 'Day_2',
                                                  2: 'Day_3'})
    
    # Cast all time columns (strings) to datetime
    room_dict[i]['Start'] = pd.to_datetime(room_dict[i]['Start'])
    room_dict[i]['End'] = pd.to_datetime(room_dict[i]['End'])
    
    # Just look at hours
    room_dict[i]['Start'] = room_dict[i]['Start'].dt.strftime('%H')
    room_dict[i]['End'] = room_dict[i]['End'].dt.strftime('%H')
    
    # Check if start and end are the same, track if they are not
    room_dict[i]['Diff'] = room_dict[i].apply(
        lambda x: None if x['Start'] == x['End'] else x['End'], axis = 1)
    
    # Impute missing values
    room_dict[i]['Diff_2'] = room_dict[i].apply(
        lambda x: int(x['End']) - 1 if int(x['End']) - int(x['Start'])
        else None, axis = 1)
    
    # Melt days
    id_vars = ['Class Description / Topic', 'Section', 'Perc_Filled',
               'Start', 'Diff', 'Diff_2']
    value_vars = [i for i in room_dict[i].columns if 'Day_' in i]
    room_dict[i] = pd.melt(room_dict[i],
                           id_vars = id_vars,
                           value_vars = value_vars).\
        rename(columns = {'value': 'Day'})
    
    # Just look at useful columns
    room_dict[i] = room_dict[i][['Perc_Filled', 'Start', 'Diff', 'Diff_2', 'Day']]
    
    # Melt start and end values
    room_dict[i] = pd.melt(room_dict[i],
                           id_vars = ['Perc_Filled', 'Day'],
                           value_vars = ['Start', 'Diff', 'Diff_2'])
    room_dict[i] = room_dict[i].drop(columns = ['variable']).\
        rename(columns = {'value': 'Hour'})
    room_dict[i] = room_dict[i].dropna()
    
    # Fill any gaps with blank data for display purposes
    fake_df = pd.DataFrame(columns = ['Hour', 'Day', 'Perc_Filled'])
    for j in range(24):
        for k in ['Mo', 'Tu', 'We', 'Th', 'Fr']:
            fake_df = fake_df.append(pd.DataFrame(
                data = {'Hour': j, 'Day': k, 'Perc_Filled': 0}, index = [0]))
    room_dict[i] = room_dict[i].append(fake_df).reset_index(drop = True)
    
    # Clean up
    room_dict[i]['Hour'] = room_dict[i]['Hour'].astype(int)
    room_dict[i]['Perc_Filled'] = room_dict[i]['Perc_Filled'].astype(float)
    room_dict[i]['Day'] = pd.Categorical(room_dict[i]['Day'],
                                         ['Mo', 'Tu', 'We', 'Th', 'Fr'])