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
import os
def find_file_list(path, substrs):
    file_list =[]
    for root, directory, files in os.walk(path):
        for f in files:
            for s in substrs:
                if s in f:
                    file_list.append(os.path.join(root, f))
    file_list.sort()
    return file_list

def read_wdcgg(path, input_file):
    df_list=[] ; flag_list=[]
    flist = find_file_list( path+input_file, ['event'] )

    df = pd.read_csv( flist[0], header=207, delim_whitespace=True, index_col=0 )
    station=df.index[0]
    df.index = pd.to_datetime( df[['year','month','day','hour','minute','second']])#, format='%Y-%m-%d %H:%M' )
    df = df[df.QCflag==1]

    df = pd.DataFrame( {'Value':df.value}, index=df.index) 
    return df, station

def main():
    path = '/mnt/lustre/users/mjr583/NCAS_CVAO/GAW_datasets/'
    input_file = 'WDCGG_20220902134435/'
    print( path, input_file )
    df, station = read_wdcgg(path, input_file)
    print( df )
    df.to_csv(f'{path}/{station}_CO.csv')

if __name__ == "__main__":
    main()
