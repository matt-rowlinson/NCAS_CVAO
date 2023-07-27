#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 11:26:20 2020

@author: matthewrowlinson
"""
import urllib.request, json 
import matplotlib.pyplot as plt
import pandas as pd
from sites_dicts import EPA_dict
import os
plt.style.use('seaborn-darkgrid')

for s in EPA_dict:
    state=EPA_dict[s]['state_code']
    county=EPA_dict[s]['county_code']    
    site=EPA_dict[s]['site_code']

    ozone=[] ; dt=[]
    for year in range(2007,2021):
        year=str(year)
        with urllib.request.urlopen("https://aqs.epa.gov/data/api/dailyData/bySite?email=matthew.rowlinson@york.ac.uk&key=duncat32&param=44201&bdate="+year+"0101&edate="+year+"1231&state="+state+"&county="+county+"&site=9991") as url:
            data = json.loads(url.read().decode())
            
        with open('data.json', 'w') as f:
            json.dump(data, f)
            
        with open('data.json', 'r') as myfile:
            data=myfile.read()
        
        os.system("rm data.json")

        # parse file
        obj = json.loads(data)  
        state_name=obj['Data'][0]['site_address'][-8:-6]
        county_name=obj['Data'][0]['county']
        for i in range(len(obj['Data'])):
            if obj['Data'][i]['pollutant_standard'] == 'Ozone 1-hour 1979':
                ozone.append(obj['Data'][i]['arithmetic_mean']*1e3)
                dt.append(obj['Data'][i]['date_local'])
        
    df=pd.DataFrame({'Daily mean ozone':ozone})
    df.index=pd.to_datetime(dt, format='%Y-%m-%d')
    df=df.resample('M').mean()
    df.to_csv('../EPA_datasets/'+EPA_dict[s]['save_name']+'_ozone.csv')

    
    f,ax=plt.subplots(figsize=(15,5))
    ax.plot(df.index,df['Daily mean ozone'], label=county_name+' '+state_name)
    ax.set_ylabel('Daily mean $O_3$ (ppbv)')
    ax.set_ylim([10,50])
    plt.legend()
    plt.savefig('../EPA_datasets/plots/'+EPA_dict[s]['save_name']+'_'+EPA_dict[s]['state_abbr']+'_daily_ozone.png')
    plt.close()
    print(s, 'done.')
