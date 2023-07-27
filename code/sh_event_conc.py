#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:24:25 2020

@author: mjr583
"""
import sys
sys.path.append('/users/mjr583/scratch/python_lib')
import RowPy as rp
import CVAO_tools as CV
from CVAO_dict import CVAO_dict as d
import matplotlib.pyplot as plt
import pandas as pd
import netCDF4
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
from sites_dicts import EPA_dict
plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = (12, 12)
gaw_path = '/users/mjr583/scratch/NCAS_CVAO/GAW_datasets/'
epa_path = '/users/mjr583/scratch/NCAS_CVAO/EPA_datasets/'

## CVAO
species=['O3','CO','NO2','NO','CH4','ethane','propane']
df_list=[] ; hms=[]
for s in species:
    df=CV.get_from_merge(d[s])
    X,Y,time=CV.remove_nan_rows(df,df.index)
    #df=pd.DataFrame({s:df})
    if s=='O3':
        df['2009-07-01' : '2009-09-30'] = np.nan
    cv_hm=df.resample('H').mean()
    
    df_list.append(df)
    hms.append(cv_hm)
df=pd.concat(df_list)

startdate='2015-08-21'
enddate='2015-09-06'
df=df[startdate:enddate]

airmass=rp.get_trajectory_pc(end=2018)['South Atlantic'][startdate:enddate]
sh=pd.DataFrame(pd.read_csv('/users/mjr583/scratch/flexpart/postprocess/flags/files/sh_flag.csv',skiprows=1,index_col=0))
sh.index=pd.to_datetime(sh.index, format='%Y-%m-%d %H:%M:%S')
sh=sh[startdate:enddate]

airmass=sh['SH']

cols=df.columns
fig,ax=rp.create_figure(len(df.columns), vertical=True,figsize=(12,12))
for a,axes in enumerate(ax):
    axes.plot(df[cols[a]], 'k', label=cols[a])
    #axes.axvspan(datetime(2017,8,29,15),datetime(2017,8,31,16), facecolor='r', alpha=.3)
    axes.axvspan(datetime(2015,8,30,15),datetime(2015,8,31,18), facecolor='r', alpha=.3)

    axes.set_ylabel(d[species[a]]['abbr']+' ('+d[species[a]]['unit']+')', )
    myF=mdates.DateFormatter('%d/%m/%y')
    axes.xaxis.set_major_formatter(myF)
    axes.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axes.legend()
    
    ax2=axes.twinx()
    ax2.plot(airmass, 'g--', alpha=.8)
    #ax2.set_ylabel('South Atlantic %') 
    ax2.set_ylabel('Southern Hemisphere %')

    ax2.grid(None)

    axes.xaxis.set_major_formatter(myF)
    axes.xaxis.set_major_locator(mdates.DayLocator(interval=2)) 

#plt.savefig('../plots/irma_concetrations.png')
plt.savefig('../plots/fred_concetrations_SH.png')
plt.close()
