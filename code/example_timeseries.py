#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
from sites_dicts import EPA_dict
plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = (12, 12)

filepath = '../CVAO_datasets/20200908_CV_merge.csv'
df=pd.read_csv(filepath, index_col=0)
df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')

acet = df['acetone']
plt.plot(acet)
plt.savefig('./test.png')
plt.close()

