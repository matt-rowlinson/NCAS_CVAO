#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#SBATCH --partition=interactive
#SBATCH --time=00:02:00
#SBATCH --mem=1gb
#SBATCH --output=LOGS/ttest_%A.log
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib
matplotlib.use('agg')
import sys
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append('/users/mjr583/scratch/python_lib')
import RowPy as rp
import CVAO_tools as CV
from CVAO_dict import CVAO_dict as d
import seaborn as sns
plt.style.use('seaborn-darkgrid')
from netCDF4 import Dataset

global options,args
import optparse
parser = optparse.OptionParser(
        formatter = optparse.TitledHelpFormatter(),
            usage = globals()['__doc__'])

parser.add_option('-s','--species',
    dest='species',
    help='CVAO species to plot')
parser.add_option('-S','--startyear',
    dest='start',
    default='2007',
    help='Year from which to plot trends')
parser.add_option('-e','--endyear',
    dest='end',
    default='2021',
    help='Year to which to plot trends')
(options,args)=parser.parse_args()

font = {'weight' : 'bold',
        'size'   : 16}

if not options.species:
    parser.error('Must give a species to plot')
species=options.species
startyear=options.start

if species=='O3' or species=='CO':
    ## Load data from GAW
    var=rp.gaw_data(species)
    var=var.resample('6h').mean()['2007':]
else:
    ## Load data from merge
    df=CV.get_from_merge(d[species], timestep='H')
    X,Y,time=CV.remove_nan_rows(df,df.index)
    df=Y['2007':]
    dff=df.resample('6h').mean()
    var=dff['mean']
    var=pd.DataFrame(var, index=dff.index)

## Load csv with air mass %
airmass=rp.get_trajectory_pc(end=2021)
traj_end=airmass.index[-1]
var_end=var.index[-1]
if var_end > traj_end:
    var=var[:traj_end]
elif traj_end > var_end:
    airmass=airmass[:var_end]
elif var_end==traj_end:
    pass

var=var.resample('M').mean()
df=var.dropna()

intro=df['2007':'2011']
end=df['2015':'2019']

print(intro)
print(end)
from scipy.stats import ttest_ind
t,p=ttest_ind(intro,end,equal_var=True)
print(t,p)

from scipy.stats import f_oneway
F,p=f_oneway(intro, end)
print(F,p)
