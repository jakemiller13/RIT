import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('poisson 1_16.txt', header = None)
df['i'] = df.index + 1
mean = df.mean().values[0]
df = df.rename(columns = {0: 'value'})

print('Mean: ' + str(mean))

df['L'] = df.apply(lambda row: math.exp(-row['i']) * row['i']**row['value'] / math.factorial(row['value']), axis = 1)
# df['L'] = df.apply(lambda row: math.exp(-row['value']) * row['value']**row['i'] / math.factorial(row['i']), axis = 1)

df2 = pd.DataFrame(df['value'].value_counts(sort = False))
df2['i'] = df2.index
df3 = df2.apply(lambda row: math.exp(-row['i']) * row['i']**row['value'] / math.factorial(row['value']), axis = 1)
