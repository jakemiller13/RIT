#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 08:02:07 2020

@author: jmiller
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

op_df = pd.read_csv('./exact/build/optimized_results/fitness_log.csv',
                    usecols = [' Best Val. MSE'])
un_df = pd.read_csv('./exact/build/fitness_log.csv',
                    header = None,
                    delimiter = ' ',
                    usecols = [1])

# Rename columns for plotting
op_df.columns = ['Optimized']
un_df.columns = ['Unoptimized']

# Start by dropping rows that can't be compared
df = un_df.join(op_df).dropna()
df.plot(title = 'Optimized (EXAMM) vs. Un-Optimized (RNN)\nAll Epochs',
        xlabel = 'Epoch',
        ylabel = 'MSE')

# Compare first iterations for easier viewing
df.iloc[0:250].plot(
    title = 'Optimized (EXAMM) vs. Un-Optimized (RNN)\n250 Epochs',
    xlabel = 'Epoch',
    ylabel = 'MSE')
