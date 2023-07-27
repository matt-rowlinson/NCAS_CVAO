#!/usr/bin/env python3

import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import sys

def find_file_list(path, substrs):
    file_list =[]
    for root, directory, files in os.walk(path):
        for f in files:
            if substrs[0] in f and substrs[1] in f:
                    file_list.append(os.path.join(root, f))
    file_list.sort()
    return file_list

def sort_times_pre2011(df):
    newlist = [word for line in df['Time;'].values for word in line.split(':')]
    hour = list(map(int, newlist[::2]))
    hour = list(map(str,[x-1 for x in hour]))
    hour = [x.zfill(2) for x in hour]

    df.index = df.index + " " + hour
    df.index = pd.to_datetime( df.index, format="%Y-%m-%d %H" )
    return df.drop(df.columns[0], axis=1)

def sort_times_post2011(df):
    df['MONTH'] = df['MON']
    df['HOUR'] = df['HR']
    df.index = pd.to_datetime(df[['YEAR','MONTH','DAY','HOUR']])
    df['Value'] = df['O3(PPB)']
    return df[['Value']]


###-------------------------------MAIN-SCRIPT---------------------------------###
def main():
    path='/mnt/lustre/users/mjr583/NCAS_CVAO/ozone_datasets/am_samoa/'
    flist = find_file_list( path, ["hour",".dat"] )
    
    
    df_=[]
    for infile in flist:
        print( infile )
        if "1975-" in infile:
            df = pd.read_csv(infile, header=24, index_col=None, delim_whitespace=True)
            df = sort_times_pre2011(df)
        else:
            df = pd.read_csv(infile, header=11, index_col=None, delim_whitespace=True)
            df = sort_times_post2011(df)
        
        df_.append(df)
    
    df = pd.concat(df_)
    df = df[df != -9999999.90]
    df = df[df != 9999999.9]
    df = df[df != 9999.99].dropna()
    df = df[df < 100.]
    
    df.to_csv('SMO_ozone_1975-2015.csv')
    
    plt.scatter( df.index, df.Value )
    plt.savefig( 'scatter_all_data.png')
    plt.close()

if __name__=="__main__":
    main()
