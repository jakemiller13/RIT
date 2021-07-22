import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

#############
# PROBLEM 1 #
#############
data_1 = {'Decade': [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980],
          'Percent Complete': [72.7, 63.4, 50.0, 44.3, 41.6,
                               32.8, 27.2, 22.5, 13.3]}
df_1 = pd.DataFrame(data_1)
df_1 = sm.add_constant(df_1)

endog_1 = df_1['Percent Complete'] / 100
exog_1 = df_1[['const', 'Decade']]

##############
# PROBLEM 1a #
##############
# Linear Probability Model
lpm_mod = sm.GLM(endog_1, exog_1,
                 family = sm.families.Binomial(sm.families.links.identity()))
lpm_res = lpm_mod.fit()
print(lpm_res.summary2())

# Logistic Regression Model
logit_mod = sm.GLM(endog_1, exog_1,
                   family = sm.families.Binomial(sm.families.links.logit()))
logit_res = logit_mod.fit()
print(logit_res.summary2())

# Probit Model
probit_mod = sm.GLM(endog_1, exog_1,
                    family = sm.families.Binomial(sm.families.links.probit()))
probit_res = probit_mod.fit()
print(probit_res.summary2())

# Ordinary Least Squares Regression
ols_mod = sm.OLS(endog_1, exog_1)
ols_res = ols_mod.fit()
print(ols_res.summary2())

##############
# PROBLEM 1b #
##############
for model in [lpm_res, logit_res, probit_res]:
    plt.plot(range(len(model.predict())),
             model.predict(),
             label = str(model.family.link).split('.')[-1].split(' ')[0])
plt.plot(range(len(ols_res.predict())), ols_res.predict(), label = 'OLS')
plt.scatter(range(len(df_1['Percent Complete'])),
            df_1['Percent Complete'] / 100)
plt.xlabel('Decade (19#0)')
plt.ylabel('Fraction of games completed')
plt.legend()
plt.text(0.1, 0.3, 'Fits are generally ok\nexcept for 1920s and 1930s')
plt.show()

##############
# PROBLEM 1c #
##############
for model in [lpm_res, logit_res, probit_res, ols_res]:
    predict = model.predict([[1, 1990], [1, 2000], [1, 2010]])
    try:
        print('--- {} ---'.format(
            str(model.family.link).split('.')[-1].split(' ')[0]))
    except AttributeError:
        print('--- OLS ---')
    [print('{}: {}'.format(year, round(predict[pred], 4)))
     for pred, year in enumerate([1990, 2000, 2010])]
    if (predict > 0).all():
        print('Plausible because yields all positive but decreasing values')
    print()


#############
# PROBLEM 2 #
#############
'''
D = duration of surgery in minutes
T = type of device (0 = laryngeal mask airway, 1 = tracheal tube)
Y = sore throat (0 = no, 1 = yes)
'''
data_2 = {'Patient': range(1, 36),
          'D': [45, 15, 40, 83, 90, 25, 35, 65, 95, 35, 75, 45,
                50, 75, 30, 25, 20, 60, 70, 30, 60, 61, 65, 15,
                20, 45, 15, 25, 15, 30, 40, 15, 135, 20, 40],
          'T': [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1,
                1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1,
                1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1],
          'Y': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0,
                0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0]}

df_2 = pd.DataFrame(data_2)
df_2 = sm.add_constant(df_2)

endog_2 = df_2['Y']
exog_2 = df_2[['const', 'D', 'T']]

##############
# PROBLEM 2a #
##############
logit_mod_2 = sm.GLM(endog_2, exog_2,
                     family = sm.families.Binomial(sm.families.links.logit()))
logit_res_2 = logit_mod_2.fit()
print(logit_res_2.summary2())
print('-- Pearson --\nValue / DoF = {} / {} = {}'.format(
    round(logit_res_2.pearson_chi2, 3),
    logit_res_2.df_resid,
    round(logit_res_2.pearson_chi2 / logit_res_2.df_resid, 3)))
print('\n-- Deviance --\nValue / DoF = {} / {} = {}'.format(
    round(logit_res_2.deviance, 3),
    logit_res_2.df_resid,
    round(logit_res_2.deviance / logit_res_2.df_resid, 3)))
print('\nBoth close to 1, therefore fit is ok')

##############
# PROBLEM 2b #
##############
df_2['Predicted'] = logit_res_2.predict()
df_2 = df_2.sort_values('D')
x_0 = df_2[df_2['T'] == 0]
x_1 = df_2[df_2['T'] == 1]

plt.plot(x_0['D'], x_0['Predicted'], label = 'laryngeal mask airway')
plt.plot(x_1['D'], x_1['Predicted'], label = 'tracheal tube')
plt.xlabel('Duration (Minutes)')
plt.ylabel('Probability')
plt.title('Problem 2b')
plt.legend()
plt.text(80, 0.6, 'As duration increases, so does\nprobability of sore throat')
plt.show()

print('As duration increases, so does probability of sore throat')

##############
# PROBLEM 2c #
##############
conf_int = logit_res_2.conf_int()
conf_int.columns = ['5%', '95%']
conf_int['Odds Ratio'] = logit_res_2.params
conf_int = np.exp(conf_int)
print(conf_int)

print('For 10 minute increase, a person is between {} and {} times as likely '
      'to get sore throat.'.format(round(conf_int['5%']['D'], 2),
                                   round(conf_int['95%']['D'], 2)))

##############
# PROBLEM 2d #
##############
print('A person who had a sore throat after surgery is between {} and {} '
      'times as likely to have received a tracheal tube instead of a '
      'laryngeal mask airway.'.format(round(conf_int['5%']['T'], 2),
                                      round(conf_int['95%']['T'], 2)))

##############
# PROBLEM 2e #
##############
sm.qqplot(logit_res_2.resid_pearson, line = 'r')
plt.title('Q-Q Plot of Pearson Residuals')
plt.show()

print('Pearson residuals:\n{}'.format(logit_res_2.resid_pearson))
print('Based on residuals and Q-Q plot, it appears there are some issues at '
      'extreme values, but middle region looks ok.')

##############
# PROBLEM 2f #
##############
print('Goodness of fit suggests this model is good. Residuals are not '
      'obviously perfect, showing some issues at extremes, but otherwise '
      'are mostly ok. Therefore this model is probably ok.')


#############
# PROBLEM 3 #
#############
df_3 = pd.read_excel('/Users/jmiller/Google Drive/RIT/'
                     'STAT.784 - Categorical Data Analysis/umaru.xlsx')

##############
# PROBLEM 3a #
##############
print('IV Drug Use History at Admission is qualitative. We can code these '
      '3 levels (1 = Never, 2 = Previous, 3 = Recent) by using 2 dummy '
      'variables.')

df_3 = df_3.join(pd.get_dummies(df_3['ivhx'],
                                prefix = 'ivhx',
                                drop_first = True))
df_3 = df_3.drop(['ivhx'], axis = 1)

##############
# PROBLEM 3b #
##############
endog_3 = df_3['dfree']
exog_3 = df_3[['age', 'beck', 'ndrugtx', 'race', 'treat', 'site', 'ivhx_2',
               'ivhx_3']]

logit_mod_3 = sm.GLM(endog_3, exog_3,
                     family = sm.families.Binomial(sm.families.links.logit()))
logit_res_3 = logit_mod_3.fit()
print(logit_res_3.summary2())

print('-- Pearson --\nValue / DoF = {} / {} = {}'.format(
    round(logit_res_3.pearson_chi2, 3),
    logit_res_3.df_resid,
    round(logit_res_3.pearson_chi2 / logit_res_3.df_resid, 3)))
print('\n-- Deviance --\nValue / DoF = {} / {} = {}'.format(
    round(logit_res_3.deviance, 3),
    logit_res_3.df_resid,
    round(logit_res_3.deviance / logit_res_3.df_resid, 3)))
print('\nBoth close to 1, therefore fit is ok')

##############
# PROBLEM 3c #
##############
print('Race parameter interval includes 0, therefore unclear how much race '
      'would contribute to final prediction. I would likely remove this '
      'variable from the final model.')

##############
# PROBLEM 3d #
##############
print('The interval for IVHX 2 vs 1 includes 0, therefore I would likely '
      'drop this from model. However, IVHX 3 vs 1 clearly has a negative '
      'impact on final prediction.')
