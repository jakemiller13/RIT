#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 18:48:02 2021

@author: jmiller
"""

import pandas as pd

bel = pd.read_csv('~/Google Drive/RIT/ISTE.782 - Visual Analytics/Week 4/' + \
                  'API_BLZ_DS2_en_csv_v2_2061686/' + \
                  'API_BLZ_DS2_en_csv_v2_2061686.csv', skiprows = 4)
eri = pd.read_csv('~/Google Drive/RIT/ISTE.782 - Visual Analytics/Week 4/' + \
                  'API_ERI_DS2_en_csv_v2_2060165/' + \
                  'API_ERI_DS2_en_csv_v2_2060165.csv', skiprows = 4)
phi = pd.read_csv('~/Google Drive/RIT/ISTE.782 - Visual Analytics/Week 4/' + \
                  'API_PHL_DS2_en_csv_v2_2059465/' + \
                  'API_PHL_DS2_en_csv_v2_2059465.csv', skiprows = 4)
usa = pd.read_csv('~/Google Drive/RIT/ISTE.782 - Visual Analytics/Week 4/' + \
                  'API_USA_DS2_en_csv_v2_2056322/' + \
                  'API_USA_DS2_en_csv_v2_2056322.csv', skiprows = 4)

indicators = ['Population ages 65 and above, total', 'Population, total']
    
bel = bel.loc[bel['Indicator Name'].isin(indicators)]
eri = eri.loc[eri['Indicator Name'].isin(indicators)]
phi = phi.loc[phi['Indicator Name'].isin(indicators)]
usa = usa.loc[usa['Indicator Name'].isin(indicators)]

years = [str(i) for i in range(1998, 2019)]

bel = bel[years].T.reset_index()
bel = bel.rename(columns = {'index': 'Year',
        bel.columns[1]: 'Population ages 65 and above, total',
        bel.columns[2]: 'Population, total'})

eri = eri[years].T.reset_index()
eri = eri.rename(columns = {'index': 'Year',
        eri.columns[1]: 'Population ages 65 and above, total',
        eri.columns[2]: 'Population, total'})

phi = phi[years].T.reset_index()
phi = phi.rename(columns = {'index': 'Year',
        phi.columns[1]: 'Population ages 65 and above, total',
        phi.columns[2]: 'Population, total'})

usa = usa[years].T.reset_index()
usa = usa.rename(columns = {'index': 'Year',
        usa.columns[1]: 'Population ages 65 and above, total',
        usa.columns[2]: 'Population, total'})

bel['Population ages 65 and above, percent'] = \
    bel['Population ages 65 and above, total'] / bel['Population, total']
eri['Population ages 65 and above, percent'] = \
    eri['Population ages 65 and above, total'] / eri['Population, total']
phi['Population ages 65 and above, percent'] = \
    phi['Population ages 65 and above, total'] / phi['Population, total']
usa['Population ages 65 and above, percent'] = \
    usa['Population ages 65 and above, total'] / usa['Population, total']

bel.insert(0, 'Country', 'Belize')
eri.insert(0, 'Country', 'Eritrea')
phi.insert(0, 'Country', 'Phillipines')
usa.insert(0, 'Country', 'United States of America')

df = bel.append(eri).append(phi).append(usa).reset_index(drop = True)
df['Year'] = pd.to_datetime(df['Year'])
df.to_csv('~/Google Drive/RIT/ISTE.782 - Visual Analytics/Week 4/' + \
          'indicators.csv')