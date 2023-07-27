#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:06:54 2020

@author: matthewrowlinson
"""
from netCDF4 import Dataset
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import datetime

path='/users/mjr583/scratch/NCAS_CVAO/CVAO_datasets/O3_tei/'
df_list=[] ; flag_list=[] ; ind_list=[]
for infile in sorted(glob.glob(path+'cv-tei*na')):
    f = open(infile)
    rows = int(f.readline()[:2])
    print(infile)
    fh = np.loadtxt(infile,skiprows=rows)
    
    #ind_list.append(pd.Series(fh[:,0]))
    df_list.append(pd.Series(fh[:,1],index=fh[:,0]))
    flag_list.append(pd.Series(fh[:,2]))
o3 = pd.concat(df_list)
flag = pd.concat(flag_list)

df = pd.DataFrame(o3, index=o3.index)
df.columns=['O3']
df['flag']=pd.Series(flag.values, index=df.index)

new_date=[]
for t, tt in enumerate(df.index):
    x = (datetime.datetime(2006,1,1,0,0) + datetime.timedelta(tt-1))
    new_date.append(x)

df.index=new_date

df = df[df.O3 != 9999.0]
df = df[df.flag != 3]
df = df[df.flag != 2]
#df = df[df.flag != 1]

df.plot()
