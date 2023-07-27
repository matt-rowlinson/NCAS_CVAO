#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import glob 
from datetime import datetime, timedelta
import numpy as np
import re
import sys
plt.style.use('seaborn-darkgrid')

def read_nas(path, input_file):
    df_list=[] ; flag_list=[]
    for infile in sorted(glob.glob(f'{path}{input_file}/*nas')):
        print(infile)
        with open(infile) as thefile:
            try:
                header= np.array([next(thefile) for x in range(90) ])
            except:
                continue
        for n,ii in enumerate(header):
            if "Startdate" in ii:
                s=int(re.search(r'\d+',ii).group())
            elif "Station name" in ii:
                station=ii.split()[-1]
            elif "Component" in ii:
                var = ii.split()[-1]
        starttime=pd.to_datetime(s, format='%Y%m%d%H%M%S')
        #print( starttime )
        #sys.exit() 
        for nskip in range(40,120): ## Find where the header ends and values begin
            try:
                fh=np.loadtxt(infile, skiprows=nskip)
                print(nskip)
                break
            except:
                continue
        df=pd.DataFrame(fh, index=fh[:,0])
        df, flag = find_times(df, starttime)

        df_list.append(df)
        flag_list.append(flag)
    
    dddf = pd.concat(df_list)
    flags=pd.concat(flag_list)
    flags.index=pd.to_datetime(flags.index, format='%Y-%m-%d %H:%M:%S')

    return dddf, flags, station, var

def find_times(df, starttime):
    t0=starttime
    endtime=df[df.columns[1]]
    
    timex=[]
    for i in range(len(endtime)):
        timex.append(t0 + timedelta(days=endtime.values[i]))
    df.index = timex
    dd=df
    df = dd[dd.columns[2]]
    flag = dd[dd.columns[-1]]
    
    return df, flag

def plot():
    f,ax = plt.subplots(figsize=(15,5))
    ax.scatter(df.index,df, color='r')
    ax.plot(mon.index,mon)
    ax.set_ylabel('Monthly mean $O_3$ (ppbv)')
    plt.savefig('/users/mjr583/INPUT-TO-OUTPUT.png')
    plt.close()
    return

def main():
    path = '/mnt/lustre/users/mjr583/NCAS_CVAO/GAW_datasets/'
    input_file = 'Ebas_221019_1700/'
    print( path, input_file )

    df, flag, station, var = read_nas(path, input_file)
    
    df = pd.DataFrame( {'Value' : df.values, 'Flag' : flag.values}, index=df.index )
    print( df )
    df=df[df.Flag == 0]
    #df = df[1:]
    df.index.name='Datetime'
    print(df)
    
    df.to_csv(f'{path}/{station}_{var}.csv')

if __name__ == "__main__":
    main()
