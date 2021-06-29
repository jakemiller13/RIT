from statsmodels.stats.contingency_tables import Table, Table2x2
import numpy as np

#############
# PROBLEM 1 #
#############
data1 = np.array([[1601, 162527], [510, 412368]])
table1 = Table2x2(data1)

print('1a: Odds Ratio = {}'.format(table1.oddsratio))
print('1b: Confidence Interval = {}'.format(table1.oddsratio_confint(0.05)))
print('1c: Expected counts = \n{}'.format(table1.fittedvalues))
print('1d: Pearson Chi-Square = {}'.format(table1.chi2_contribs.sum()))
print()

#############
# PROBLEM 2 #
#############
data2 = np.array([[103, 15, 11], [341, 105, 405]])
table2 = Table(data2)
table2_2 = Table(data2[:, 0: 2])
table2_3 = Table(np.array([data2[:, 0] + data2[:, 1], data2[:, 2]]))
table2_4 = Table2x2([data2[:, 0], data2[:, 2]])

print('2a: \n{}'.format(table2.test_nominal_association()))
print('Small p-value, therefore party and race are associated\n')
print('2b: \n{}'.format(table2.resid_pearson))
print('Strong associations for both democrat and republican. '
      'Minimal association for independent.\n')
print('2c:')
print('W/o Repub\n----------\n{}\n\nDem + Ind\n----------\n{}'.format(
      table2_2.test_nominal_association(),
      table2_3.test_nominal_association()))
print('Similar outcome with either removal of Repub or combined Dem + Indep\n')
print('2d: {}'.format(table2_4.oddsratio_confint(0.05)))
print('Significantly higher odds (6 - 21 times) that Democrat is Black '
      'rather than White')
print()

#############
# PROBLEM 3 #
#############
print('3: Small p-values, therefore reject null - there is an association '
      'between gender and political party.')
print()

#############
# PROBLEM 4 #
#############
data4 = np.array([[26, 51, 21, 40], [31, 59, 11, 34]])
table4 = Table(data4)

print('4a:\n{}\n'.format(table4.test_nominal_association()))
print('4b:\n{}\n'.format(table4.test_ordinal_association(
      col_scores = np.array([3, 2, 1, 0]))))
print('4c:\n{}\n'.format(table4.test_ordinal_association(
      col_scores = np.array([3.5, 2.5, 1.5, 0.5]))))
print('4d:\n{}\n'.format(table4.test_ordinal_association()))
print('4e: All show large p-values, therefore no association.')
