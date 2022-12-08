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
