import pandas as pd
import numpy as np
import statsmodels.api as sm


def pprint(text):
    length = len(text)
    print()
    print((length + 4) * '#')
    print('# {} #'.format(text))
    print((length + 4) * '#')


data = {'V-R':   [116, 16, 2, 162],
        'V-NR':  [54, 24, 8, 18],
        'NV-R':  [324, 104, 18, 18],
        'NV-NR': [106, 156, 72, 2]}

df = pd.DataFrame.from_dict(data)
df.index = ['WM', 'WF', 'BM', 'BF']

pprint('** NOTE FOR TABLES BELOW **')
print('Mantel-Haenszel = "Test of OR=1"')
print('Breslow-Day = "Test of constant OR"')

##############
# PROBLEM 1a #
##############
df_a = pd.concat([df.iloc[0] + df.iloc[2],
                  df.iloc[1] + df.iloc[3]], axis = 1).T
df_a.index = ['Male', 'Female']

tables_a = [df_a[['V-R', 'V-NR']], df_a[['NV-R', 'NV-NR']]]
st_a = sm.stats.StratifiedTable(tables_a)

pprint('Problem 1a')
print(pd.concat([tables_a[0], tables_a[1]], axis = 1))
print()
print(st_a.summary())
print('Based on small p-value, there is an interaction.')

##############
# PROBLEM 1b #
##############
df_b = pd.concat([df.iloc[0:2].sum(), df.iloc[2:4].sum()], axis = 1).T
df_b.index = ['White', 'Black']

tables_b = [df_b[['V-R', 'V-NR']], df_b[['NV-R', 'NV-NR']]]
st_b = sm.stats.StratifiedTable(tables_b)

pprint('Problem 1b')
print(pd.concat([tables_b[0], tables_b[1]], axis = 1))
print()
print(st_b.summary())
print('Based on small p-value, there is an interaction.')

##############
# PROBLEM 1c #
##############
arr = np.array(df)

tables_c = [np.reshape(i, (2, 2)) for i in arr]
st_c = sm.stats.StratifiedTable(tables_c)

pprint('Problem 1c')
print(st_c.summary())
print('Based on very high p-value, there is no interaction when controlling '
      'for both race and gender.')

##############
# PROBLEM 1d #
##############
pprint('Problem 1d')
print('Both race and gender, independently, have effects on vaccinations. '
      'However, when taken together, they do not.\nThis assessment is '
      'based on the interactions seen in parts (a) and (b), and the lack of '
      'interaction in part (c).')