#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:54:57 2020

@author: mjr583
"""
import sys
sys.path.append('/users/mjr583/python_lib')
import CVAO_tools as CV
from CVAO_dict import CVAO_dict as d

df  = CV.get_from_merge(d['O3'], timestep='M')
X, Y, time = CV.remove_nan_rows(df, df.index)

#CV.plot_trend_with_func_from_dict(d, force_merge=True, timestep='H')
CV.plot_trend_with_func_from_dict(d, force_merge=True, timestep='M')
