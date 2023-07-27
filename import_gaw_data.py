#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:09:25 2020

@author: matthewrowlinson
"""
#

def import_gaw_data(url):
    import urllib
    import pandas as pd
    urllib.request.urlretrieve('ftp://aftp.cmdl.noaa.gov/data/ozwv/SurfaceOzone/PCO/pco_o3_ppb_2001-2015.dat', '/Users/matthewrowlinson/Documents/file.dat')
    file = '/Users/matthewrowlinson/Documents/file.dat'
    df=pd.read_csv(file, header=22, delimiter=';', index_col=0)
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d  %H:%M').strftime('%Y/%m/%d %H:%M')
    df.columns = ['Value']
    df = df[df.Value != -999.0]
    return df