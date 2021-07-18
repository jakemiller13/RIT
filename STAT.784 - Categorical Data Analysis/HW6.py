import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from math import ceil, floor

alpha = 0.05

#############
# PROBLEM 1 #
#############
print('*** PROBLEM 1 ***')

data = {'Temp': [7, 5, 7, 7, 6, 6, 5, 6, 5, 6, 5, 7, 6, 6, 6],
        'Oil': [4, 3, 3, 2, 4, 3, 3, 2, 4, 2, 2, 3, 3, 3, 4],
        'Time': [90, 105, 105, 90, 105, 90, 75, 105, 90, 75, 90, 75, 90, 90, 75],
        'y': [24, 28, 40, 42, 11, 16, 126, 34, 32, 32, 34, 17, 30, 17, 50]}

df = pd.DataFrame(data)
df = sm.add_constant(df)

##############
# PROBLEM 1a #
##############
df_exog = df[['const', 'Temp', 'Oil', 'Time']]
df_exog = pd.concat([df_exog,
                     df_exog['Temp'] * df_exog['Oil'],
                     df_exog['Temp'] * df_exog['Time'],
                     df_exog['Oil'] * df_exog['Time']], axis = 1)
df_exog = df_exog.rename(columns = {0: 'Temp * Oil',
                                    1: 'Temp * Time',
                                    2: 'Oil * Time'})
df_endog = df['y']

model_poisson = sm.GLM(df_endog, df_exog,
                       family = sm.families.Poisson(sm.families.links.log()),
                       alpha = 0.05)
res_poisson = model_poisson.fit()
print(res_poisson.summary2())

print('Remove any variables where CI includes 0 and p > 0.05, '
      'i.e. "Temp * Oil"\n')

df_exog_2 = df_exog[[i for i in df_exog.columns if i != 'Temp * Oil']]
model_reduced = sm.GLM(df_endog, df_exog_2,
                       family = sm.families.Poisson(sm.families.links.log()),
                       alpha = 0.05)
res_reduced = model_reduced.fit()
print(res_reduced.summary2())

print('Remaining variables all show significant impact.\n')

##############
# PROBLEM 1b #
##############
disp = res_reduced.deviance / res_reduced.df_model
print('Dispersion (D / df) = {}'.format(round(disp, 2)))
print('Therefore, likely overdispersion.\n')

##############
# PROBLEM 1c #
##############
model_poi_id = sm.GLM(df_endog, df_exog_2,
                      family = sm.families.Poisson(sm.families.links.identity()),
                      alpha = 0.05)
res_poi_id = model_poi_id.fit()
print(res_poi_id.summary2())

print('AIC and BIC are lower for log link, therefore use that.')

###########################
# Q-Q AND HISTOGRAM PLOTS #
###########################
for model in [res_reduced, res_poi_id]:
    model_family = str(model.model.family).split('.')[-1].split(' ')[0]
    model_link = str(model.model.family.link).split('.')[-1].split(' ')[0]

    fig, ax = plt.subplots(2, 2)

    sm.qqplot(model.resid_pearson, line = 'r', ax = ax[0, 0])
    ax[0, 0].set_title('Q-Q Plot of Pearson Residuals')

    sm.qqplot(model.resid_deviance, line = 'r', ax = ax[0, 1])
    ax[0, 1].set_title('Q-Q Plot of Deviance Residuals')

    pearson_min_bin = floor(min(model.resid_pearson))
    pearson_max_bin = ceil(max(model.resid_pearson)) + 1
    ax[1, 0].hist(model.resid_pearson,
                  bins = range(pearson_min_bin, pearson_max_bin),
                  rwidth = 0.8,
                  align = 'left')
    ax[1, 0].set_xticks([i for i in range(pearson_min_bin, pearson_max_bin)
                         if i % 2 == 0])
    ax[1, 0].set_title('Histogram of Pearson Residuals')

    deviance_min_bin = floor(min(model.resid_deviance))
    deviance_max_bin = ceil(max(model.resid_deviance)) + 1
    ax[1, 1].hist(model.resid_deviance,
                  bins = range(deviance_min_bin, deviance_max_bin),
                  rwidth = 0.8,
                  align = 'left')
    ax[1, 1].set_xticks([i for i in range(deviance_min_bin, deviance_max_bin)
                         if i % 2 == 0])
    ax[1, 1].set_title('Histogram of Deviance Residuals')

    fig_title = 'Family: {} | Link: {}'.format(model_family, model_link)
    fig.suptitle(fig_title)
    plt.tight_layout()
    plt.show()

    ##########################
    # RESIDUALS VS VARIABLES #
    ##########################
    fig, ax = plt.subplots(2, 2)

    sorted_temp = sorted(list(zip(df['Temp'], model.resid_pearson)))
    sorted_temp_x = [i[0] for i in sorted_temp]
    sorted_temp_y = [i[1] for i in sorted_temp]

    ax[0, 0].scatter(sorted_temp_x, sorted_temp_y)
    ax[0, 0].set_title('Residuals vs. Temp')

    sorted_oil = sorted(list(zip(df['Oil'], model.resid_pearson)))
    sorted_oil_x = [i[0] for i in sorted_oil]
    sorted_oil_y = [i[1] for i in sorted_oil]

    ax[0, 1].scatter(sorted_oil_x, sorted_oil_y)
    ax[0, 1].set_title('Residuals vs. Oil')

    sorted_time = sorted(list(zip(df['Time'], model.resid_pearson)))
    sorted_time_x = [i[0] for i in sorted_time]
    sorted_time_y = [i[1] for i in sorted_time]

    ax[1, 0].scatter(sorted_time_x, sorted_time_y)
    ax[1, 0].set_title('Residuals vs. Time')

    fig_title = 'Family: {} | Link: {}'.format(model_family, model_link)
    fig.suptitle(fig_title)
    plt.tight_layout()
    plt.show()

print('Nothing stands out as abnormal in any of these plots, therefore either '
      'model likely ok to use for this analysis.')

#############
# PROBLEM 2 #
#############
print('*** PROBLEM 2 ***')

data = {'bath_temperature': [248, 248, 248, 248, 248, 248, 248, 248, 248,
                             252, 252, 252, 252, 252, 252, 252, 252, 252],
        'wave_height': [4.38, 4.38, 4.38, 4.4, 4.4, 4.4, 4.42, 4.42, 4.42,
                        4.38, 4.38, 4.38, 4.4, 4.4, 4.4, 4.42, 4.42, 4.42],
        'overhead_preheater': [340, 360, 380, 340, 360, 380, 360, 380, 380,
                               340, 360, 380, 340, 360, 380, 340, 360, 380],
        'preheater_1': [340, 360, 380, 360, 380, 340, 340, 360, 340,
                        380, 340, 360, 360, 380, 340, 380, 340, 360],
        'preheater_2': [340, 360, 380, 360, 380, 340, 340, 360, 380,
                        380, 340, 360, 380, 340, 360, 360, 380, 340],
        'air_knife': [0, 3, 6, 3, 6, 0, 6, 0, 3, 3, 6, 0, 0, 3, 6, 6, 0, 3],
        'overhead_vibration': [0, 2, 4, 4, 0, 2, 2, 4, 0,
                               2, 4, 0, 4, 0, 2, 0, 2, 4],
        'y': [4, 2, 1, 2, 6, 15, 9, 5, 8, 5, 4, 11, 10, 15, 4, 12, 6, 7]}

df = pd.DataFrame(data)
df = sm.add_constant(df)

##############
# PROBLEM 2a #
##############
df_exog = df[['const', 'bath_temperature', 'wave_height', 'overhead_preheater',
              'preheater_1', 'preheater_2', 'air_knife', 'overhead_vibration']]
df_endog = df['y']

model_glm = sm.GLM(df_endog, df_exog,
                   family = sm.families.Gaussian(sm.families.links.identity()),
                   alpha = 0.05)
res_glm = model_glm.fit()
print(res_glm.summary2())

model_poisson = sm.GLM(df_endog, df_exog,
                       family = sm.families.Poisson(sm.families.links.identity()),
                       alpha = 0.05)
res_poisson = model_poisson.fit()
print(res_poisson.summary2())

print('Remove any variables where CI includes 0 and p > 0.05, '
      'i.e. overhead_preheater, preheater_1, preheater_2\n')

df_exog_reduced = df_exog[['const', 'bath_temperature', 'wave_height',
                           'air_knife', 'overhead_vibration']]
model_poi_reduced = sm.GLM(df_endog, df_exog_reduced,
                           family = sm.families.Poisson(
                               sm.families.links.identity()),
                           alpha = 0.05)
res_poi_reduced = model_poi_reduced.fit()
print(res_poi_reduced.summary2())

print('Both Poisson are similar in AIC/BIC, with reduced Poisson being '
      'slightly better performance. Normal distribution much larger.\n')

for i in [res_glm, res_poisson, res_poi_reduced]:
    if i == res_glm:
        family = 'GLM'
    elif i == res_poisson:
        family = 'Poisson'
    elif i == res_poi_reduced:
        family = 'Reduced Poisson'

    param = round(i.params['wave_height'], 2)
    conf_int_lo = round(i.conf_int().loc['wave_height'][0], 2)
    conf_int_hi = round(i.conf_int().loc['wave_height'][1], 2)
    p_val = round(i.pvalues['wave_height'], 5)
    print('-- {} --\nParam: {} | C Int: [{} {}] | p: {}'.
          format(family,
                 param,
                 conf_int_lo,
                 conf_int_hi,
                 p_val))
    if (conf_int_lo < 0 and conf_int_hi > 0) or p_val > alpha:
        sig = 'Not Significant'
    else:
        sig = 'Significant'
    print('"b" value is: {}\n'.format(sig))

###########################
# Q-Q AND HISTOGRAM PLOTS #
###########################
model_family = 'Reduced Poisson'
model_link = 'Identity'

fig, ax = plt.subplots(2, 2)

sm.qqplot(res_poi_reduced.resid_pearson, line = 'r', ax = ax[0, 0])
ax[0, 0].set_title('Q-Q Plot of Pearson Residuals')

sm.qqplot(res_poi_reduced.resid_deviance, line = 'r', ax = ax[0, 1])
ax[0, 1].set_title('Q-Q Plot of Deviance Residuals')

pearson_min_bin = floor(min(res_poi_reduced.resid_pearson))
pearson_max_bin = ceil(max(res_poi_reduced.resid_pearson)) + 1
ax[1, 0].hist(res_poi_reduced.resid_pearson,
              bins = range(pearson_min_bin, pearson_max_bin),
              rwidth = 0.8,
              align = 'left')
ax[1, 0].set_xticks([i for i in range(pearson_min_bin, pearson_max_bin)
                     if i % 2 == 0])
ax[1, 0].set_title('Histogram of Pearson Residuals')

deviance_min_bin = floor(min(res_poi_reduced.resid_deviance))
deviance_max_bin = ceil(max(res_poi_reduced.resid_deviance)) + 1
ax[1, 1].hist(res_poi_reduced.resid_deviance,
              bins = range(deviance_min_bin, deviance_max_bin),
              rwidth = 0.8,
              align = 'left')
ax[1, 1].set_xticks([i for i in range(deviance_min_bin, deviance_max_bin)
                     if i % 2 == 0])
ax[1, 1].set_title('Histogram of Deviance Residuals')

fig_title = 'Family: {} | Link: {}'.format(model_family, model_link)
fig.suptitle(fig_title)
plt.tight_layout()
plt.show()

##########################
# RESIDUALS VS VARIABLES #
##########################
fig, ax = plt.subplots(2, 2)

sorted_wave_height = sorted(list(zip(df['wave_height'],
                                     res_poi_reduced.resid_pearson)))
sorted_wave_height_x = [i[0] for i in sorted_wave_height]
sorted_wave_height_y = [i[1] for i in sorted_wave_height]

ax[0, 0].scatter(sorted_wave_height_x, sorted_wave_height_y)
ax[0, 0].set_title('Residuals vs. Wave Height')

sorted_preheater_2 = sorted(list(zip(df['preheater_2'],
                                     res_poi_reduced.resid_pearson)))
sorted_preheater_2_x = [i[0] for i in sorted_preheater_2]
sorted_preheater_2_y = [i[1] for i in sorted_preheater_2]

ax[0, 1].scatter(sorted_preheater_2_x, sorted_preheater_2_y)
ax[0, 1].set_title('Residuals vs. Preheater 2')

sorted_air_knife = sorted(list(zip(df['air_knife'],
                                   res_poi_reduced.resid_pearson)))
sorted_air_knife_x = [i[0] for i in sorted_air_knife]
sorted_air_knife_y = [i[1] for i in sorted_air_knife]

ax[1, 0].scatter(sorted_air_knife_x, sorted_air_knife_y)
ax[1, 0].set_title('Residuals vs. Air Knife')

sorted_overhead_vibration = sorted(list(zip(df['overhead_vibration'],
                                            res_poi_reduced.resid_pearson)))
sorted_overhead_vibration_x = [i[0] for i in sorted_overhead_vibration]
sorted_overhead_vibration_y = [i[1] for i in sorted_overhead_vibration]

ax[1, 1].scatter(sorted_overhead_vibration_x, sorted_overhead_vibration_y)
ax[1, 1].set_title('Residuals vs. Overhead Vibration')

fig_title = 'P2 Family: {} | Link: {}'.format(model_family, model_link)
fig.suptitle(fig_title)
plt.tight_layout()
plt.show()

###########################
# PARAMETER MEAN RESPONSE #
###########################
for i in df.columns:
    if i in ['const', 'y']:
        continue
    means = df.groupby([i]).mean()['y']
    print('-- {} --\nMinimize at: {}'.format(i, means.idxmin()))
    print()
