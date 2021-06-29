import math
import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import Table
from scipy.stats import chi2


def pprint(text):
    length = len(text)
    print()
    print((length + 4) * '#')
    print('# {} #'.format(text))
    print((length + 4) * '#')


#############
# PROBLEM 1 #
#############
pprint('Problem 1.a')
avg_tix_per_month = 512
weeks_per_month = 4
expected_tix_per_week = avg_tix_per_month / weeks_per_month
print('Expected tickets per week: {}'.format(int(expected_tix_per_week)))

pprint('Problem 1.b')
lam = expected_tix_per_week
total = sum([math.exp(-lam) * lam**i / math.factorial(i)
             for i in range(1, 130)])
print('Probability 130+ tickets per week: {}'.format(np.round(1 - total, 3)))

pprint('Problem 1.c')
stdev = 2
observed_tix_per_week = 130
z = (observed_tix_per_week - expected_tix_per_week) / stdev
crit_z = chi2.ppf(1 - 0.05, 2)

print('H_0: lambda = 128')
print('H_a: lambda > 128')
print('z = {} | critical z = {}'.format(z, np.round(crit_z, 3)))
print('z is not greater than critical z, therefore not enough evidence to '
      'reject H_0.')


#############
# PROBLEM 4 #
#############
pprint('Problem 4')
data4 = np.array([[28, 50, 35, 47],
                  [24, 11,  7,  8],
                  [23, 37, 14,  7]])
df4 = pd.DataFrame(data = data4,
                   columns = ['Air', 'Bus', 'Car', 'Train'],
                   index = ['NY', 'NJ', 'CT'])
df4_2 = pd.DataFrame(data = [df4['Air'],
                             df4['Bus'] + df4['Car'] + df4['Train']],
                     index = ['Air', 'Land']).T
table4_1 = Table(df4)
table4_2 = Table(df4_2)

print('Original:\n{}'.format(table4_1.table_orig))
print('\nThe analyst would like to understand the relationship between state '
      'and land vs. air, therefore I will combine the various land travel '
      'methods')
print('\nModified:\n{}'.format(table4_2.table_orig))

print('\nIndependence:\n{}\nSmall p, therefore there is an association between '
      'state and method of travel'.format(table4_2.test_nominal_association()))

print('\nPearson Residuals:\n{}\nLarge positive residual for NJ by air, '
      'large negative residual for NY by air.\nSmaller positive residual for '
      'NY by land, smaller negative residual for NJ by land.\nMinimal '
      'residuals for CT by either air or land.\nTherefore, less people '
      'from New York and more people from New Jersey travelled by air than '
      'the hypothesis of independence predicts.\nSimilar for more people '
      'from New York traveling by land and less people from New Jersey.'
      .format(table4_2.resid_pearson))

print('\nChi-squared: {}\nWith df = 2 and standard deviation of 2, this is '
      'well beyond right-hand tail (chi-squared for 2 degrees of freedom at '
      'alpha(0.05) = 5.99), therefore further evidence of association.'
      .format(round(table4_2.chi2_contribs.sum().sum(), 2)))

print('\n-- Recommendation --')
print('The problem does not specify where the convention takes place, but '
      'based on the residuals, I will assume it takes place in New York.\n'
      'This is based on the higher than expected land travel for people from '
      'New York, and higher than expected air travel for people from New '
      'Jersey.\nI would therefore recommend that people traveling from New '
      'Jersey fly, people from New York travel by land, and folks from '
      'Connecticut can travel however they see fit.')
