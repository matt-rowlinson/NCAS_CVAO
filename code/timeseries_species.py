#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:54:57 2020

@author: mjr583
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.offsetbox import AnchoredText
import netCDF4
import datetime     
import os
from pylab import rcParams
import sys 
sys.path.append('/users/mjr583/scratch/python_lib')
import CVAO_tools as CV
from CVAO_dict import CVAO_dict as d

rcParams['figure.figsize'] = 12,4

if len(sys.argv) == 1:
    for D in d:
        print(D)
        try:
            CV.get_timeseries(d[D])
        except:
            print(D, ' failed.')

else:
    for D in sys.argv[1:]:
        #species=sys.argv[1]
        print(D)
        CV.get_timeseries(d[D])
