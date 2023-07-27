#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:16:18 2020

@author: mjr583
"""

import CVAO_tools as CV
from CVAO_met_dict import MET_dict as d

d = {
     'RH_10' : {'variable' : 'RH_10',   
                 'longname' : 'Relative Humidity at 10m',
                 'unit': '%',
                 'start_year' : '2006',
                 'yscale' : 'linear',
                 'negs' : 'No',
                 'merge_name' : 'RH_10M (%)',
                 'instrument' : ''
                 }
     }

CV.met_trends_with_func_from_dict(d, timestep='H')