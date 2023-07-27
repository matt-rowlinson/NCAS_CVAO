
import pandas as pd
from netCDF4 import Dataset
import datetime
from dateutil.relativedelta import relativedelta

infile = '/mnt/lustre/users/mjr583/NCAS_CVAO/GAW_datasets/co_mhd_surface-flask_2_3001-9999_monthly.nc'

fh =  Dataset(infile)
time = fh.variables['time']
co = fh.variables['value']
flag = fh.variables['QCflag']

x = datetime.datetime(1991, 6, 1, 0, 0, 0)
times=[]
for t in time:
    dt = x + relativedelta(months=t)
    times.append( dt )
df = pd.DataFrame({'CO' : co, 'flag' : flag},index=times)
df.to_csv('/mnt/lustre/users/mjr583/NCAS_CVAO/GAW_datasets/mc_CO_1991_to_2020.csv')
