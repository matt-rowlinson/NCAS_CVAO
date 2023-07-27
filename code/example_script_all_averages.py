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
''' EXAMPLE DICTIONARY '''
'''
d = {
     
     'O3' : {'variable' : 'O3',   
                 'ceda_url' : 'http://dap.ceda.ac.uk/thredds/dodsC/badc/capeverde/data/cv-tei-o3/2019/ncas-tei-49i-1_cvao_20190101_o3-concentration_v1.nc',
                 'ebas_url' : 'http://thredds.nilu.no/thredds/dodsC/ebas/CV0001G.20061002000000.20190425081904.uv_abs.ozone.air.12y.1h.GB12L_CVO_Ozone_Thermo49series.GB12L_Thermo.lev2.nc',
                 'ebas_var_name' : 'ozone_nmol_per_mol_amean',
                 'ceda_var_name' : '',               
                 'longname' : 'ozone',
                 'abbr' : '$O_3$',
                 'unit': 'ppbv',
                 'scale' : '1e9',
                 'yscale' : 'linear',
                 'start_year' : '2006',
                 'merge_name' : 'O3',
                 'instrument' : ''
                }
     }
'''

for D in d:
    print(D)
    try:
        CV.get_percentiles(d[D])
    except:
        print(D, ' failed.')
    
