#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-darkgrid')

def simple_plot(df):
    plt.scatter( df.index, df.Value, c='#a6cee3', alpha=.02)
    plt.plot(df.Value.resample('M').mean(), marker='o', c='#1f78b4')
    plt.savefig('/mnt/lustre/users/mjr583/NCAS_CVAO/CO_trends/simple_CO.png')
    plt.close()
    return

def seasonality(df):
    _=[] ; i=[]
    years=['2014','2015','2016','2017','2018','2019','2020','2021']
    for year in years:
        dfy=df.copy()
        dfy = dfy.loc[['2016','2017','2018','2019','2020']]
        #dfy.loc['2014'] = np.nan
        #dfy.loc['2015'] = np.nan
        #dfy.loc['2021'] = np.nan

        df1=dfy.groupby(dfy.index.dayofyear).mean()
        df25=dfy.groupby(dfy.index.dayofyear).quantile(.25)
        df75=dfy.groupby(dfy.index.dayofyear).quantile(.75)

        if int(year) % 4 != 0:
            df1=df1.drop(59) ; df25=df25.drop(59) ; df75=df75.drop(59)

        plt.plot( df.loc[year].resample('D').mean().index, df1.Value, label='2008-2021 mean', alpha=.4, c='grey')
        plt.fill_between( df.loc[year].resample('D').mean().index, df25.Value, df75.Value, alpha=.2, color='grey')
        plt.plot( df.loc[year].resample('D').mean().Value, label=year)
        
        plt.ylabel( 'CO (ppb)')
        plt.legend()
        plt.ylim( 60, 150)
        plt.savefig(f'/mnt/lustre/users/mjr583/NCAS_CVAO/CO_trends/Sel.seas_CO_{year}.png')
        plt.close()
        i.append(df.loc[year].resample('M').mean().index)
        _.append(df.loc[year].resample('M').mean().Value)
    
    df1=df.groupby(df.index.month).mean()
    plt.plot( df1.index, df1.Value, label='2008-2021 mean', alpha=.4, c='grey')
    for n,p in enumerate(_):
        plt.plot( df1.index, p, label=years[n])
    
    plt.ylabel( 'CO (ppb)')
    plt.legend()
    plt.savefig(f'/mnt/lustre/users/mjr583/NCAS_CVAO/CO_trends/Sel.seas_CO_all.png')
    plt.close()
     
    return

def main():
    path='/mnt/lustre/users/mjr583/NCAS_CVAO/GAW_datasets/cvao_CO.csv'
    df = pd.read_csv( path, index_col=0 )
    df.index=pd.to_datetime(df.index, format=('%Y-%m-%d %H:%M:%S'))
    df=df[df.Flag==0.0]

    simple_plot(df)
    seasonality(df)
    return



if __name__=="__main__":
    main()
