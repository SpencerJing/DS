# -*- coding = utf-8 -*-
# @Time : 2022/12/8 15:10
# @Author : Spencer
# @File : T_test.py
# @Software : PyCharm
import pandas as pd
from scipy import stats
from scipy.stats import t
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.power as smp

data = pd.read_csv('ab_data.csv')


#  clean data
def check_mixed_assignment(df):
    df1 = df[['user_id', 'group']].groupby(['user_id']).nunique().reset_index()
    # count the unique number of groups that a user was assigned to
    df2 = df1[df1['group'] > 1]['user_id'].count()
    # count the number of users assigned to both groups
    print(df2)


def remove_mixed_assignment(df):
    df1 = df[['user_id', 'group']].groupby(['user_id']).nunique().reset_index()
    # count the unique number of groups that a user was assigned to
    df2 = pd.merge(df, df1, on=['user_id'], how='left')
    return df2[df2['group_y'] == 1][['user_id', 'timestamp', 'group_x', 'landing_page', 'converted']]\
        .rename(columns={'group_x':'group'})
    # only return users assigned to either treatment or control


data1 = remove_mixed_assignment(data)


def check_exposure_bugs(df):
    print(df[(df['group'] == 'control')&(df['landing_page'] == 'new_page')]['user_id'].count())
    # count the number of users in control expposed to treatment
    print(df[(df['group'] == 'treatment')&(df['landing_page'] == 'old_page')]['user_id'].count())
    # count the number of users in treatment expposed to control


def remove_exposure_bugs(df):
    df1 = df[(df['group'] == 'control')&(df['landing_page'] == 'new_page')][['user_id', 'group']]
    # identify the users in control expposed to treatment
    df2 = df[(df['group'] == 'treatment')&(df['landing_page'] == 'old_page')][['user_id', 'group']]
    # identify the users in treatment expposed to control
    df3 = pd.concat([df1, df2])
    df4 = pd.merge(df, df3, on=['user_id'], how='left')
    return df4[df4['group_y'].isna()][['user_id', 'timestamp', 'group_x', 'landing_page', 'converted']]\
        .rename(columns={'group_x':'group'})
    # only return users with the correct exposure


data2 = remove_exposure_bugs(data1)


def check_multiple_exposures(df):
    df1 = df[['user_id', 'group']].groupby(['user_id']).count().reset_index()
    # count the number of exposures that a user had
    df2 = df1[df1['group'] > 1]['user_id'].count()
    # count the number of users that had multiple exposures
    print(df2)


def consolidate_multiple_exposures(df):
    df1 = df.groupby(['user_id', 'group', 'landing_page'])\
        .agg({'timestamp': ['min', 'max'], 'converted': ['count', 'sum']})
    # get the timestamps of the first and last exposure, the number of exposures and the number of conversions
    df1.columns = df1.columns.droplevel(0)
    df2 = df1.reset_index()
    df2['converted'] = df2.apply(lambda x: int(x['sum'] > 0), axis=1) # 1 if the user has one conversion
    df2['conversion_rate'] = 1.0*df2['sum']/df2['count'] # the number of conversions divided by the number of exposures
    return df2
    # one user will only have one row


data3 = consolidate_multiple_exposures(data2)


# sanity check
def check_any_assignment_imbalance(df):
    df1 = df[['user_id', 'group']].groupby(['group']).count().reset_index()
    # count the number of users in treatment vs. control
    print(df1)
    pvalue = stats.binom_test(df1[df1['group'] == 'treatment']['user_id'].values[0],
                              n=df1['user_id'].sum(),
                              p=0.5,
                              alternative='two-sided')
    # test whether the treatment vs. control has the equal sample size
    print(pvalue)


#  hypothesis testing
def calculate_pvalue(df):
    n_treatment = df[df['group'] == 'treatment']['user_id'].count()  # the number of users in treatment
    n_control = df[df['group'] == 'control']['user_id'].count()  # the number of users in control

    p_treatment = 1.0 * df[df['group'] == 'treatment']['converted'].sum() / n_treatment
    # the probability of a user in treatment to convert
    p_control = 1.0 * df[df['group'] == 'control']['converted'].sum() / n_control
    # the probability of a user in control to convert

    var_treatment = p_treatment * (1 - p_treatment)  # the variance of the probability of a user in treatment to convert
    var_control = p_control * (1 - p_control)  # the variance of the probability of a user in treatment to convert

    p_delta = p_treatment - p_control  # the delta of the probability of a user to convert in treatment vs. control
    print(p_delta)

    pooled_se = np.sqrt(
        var_treatment / n_treatment + var_control / n_control)  # the pooled standard error of the t test
    t_statistic = p_delta / pooled_se  # the t statistic
    dof = (var_treatment / n_treatment + var_control / n_control) ** 2 \
          / (var_treatment ** 2 / (n_treatment ** 2 * (n_treatment - 1)) + var_control ** 2 / (
                n_control ** 2 * (n_control - 1)))
    # the degree of freedom
    pvalue = 2 * t.cdf(-abs(t_statistic), dof)  # the p value of the t test
    print(pvalue)

    lower = p_delta - t.ppf(0.975, dof) * pooled_se  # the lower bound of the confidence interval
    upper = p_delta + t.ppf(0.975, dof) * pooled_se  # the upper bound of the confidence interval
    print(lower)
    print(upper)


calculate_pvalue(data3)


#  power calculation
def calculate_mde(df):
    n_treatment = df[df['group'] == 'treatment']['user_id'].count()  # the number of users in treatment
    n_control = df[df['group'] == 'control']['user_id'].count()  # the number of users in control

    power_analysis = smp.TTestIndPower()

    effect_size = power_analysis.solve_power(
        nobs1 = n_control, power = 0.8, alpha = 0.05, ratio = 1.0 * n_treatment / n_control, alternative = 'two-sided'
    )
    p = 1.0 * df['converted'].sum() / df['user_id'].count()  # the probability of a user to convert
    sd = np.sqrt(p * (1 - p))  # the standard deviation of a user to convert
    mde = effect_size * sd  # the minimum detectable effect with the current sample size
    print(mde)

    p_treatment = 1.0 * df[df['group'] == 'treatment']['converted'].sum() / n_treatment
    # the probability of a user in treatment to convert
    p_control = 1.0 * df[df['group'] == 'control']['converted'].sum() / n_control
    # the probability of a user in control to convert
    p_delta = p_treatment - p_control
    # the measured delta of the probability of a user to convert in treatment vs. control
    nobs1 = power_analysis.solve_power(
        effect_size = 1.0 * p_delta / sd, power = 0.8, alpha = 0.05, ratio = 1.0, alternative = 'two-sided'
    )
    # the required sample size when setting the currently measured delta as the minimum detectable effect
    print(nobs1)


calculate_mde(data3)


#  validation
X = pd.get_dummies(data3['group'], drop_first=True)
y = data3['converted']

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())