import numpy as np
import xarray as xr
import pandas as pd
import os
from dask.diagnostics import ProgressBar

output_path = "E:\CE_DATA\Data_Processing\Average_Mode\R95P\CMIP6\PR\\"  # 输出结果所在文件夹
# Type=['CES','CET','CETS']
Type=['PR']
SSP=['Historical','126','245','585']
Mode=['BCC-CSM2-MR','CanESM5','CMCC-ESM2','GFDL-ESM4','IPSL-CM6A-LR','MIROC6','NESM3']
filePath='F:\\'
# Variable=['R95P','R95D','R95DM',]
Variable=['R95D','R95DM']
for ty in range(len(Type)):
    for v in range(len(Variable)):
        for s in range(len(SSP)):
            Files_name=[filePath+F'{Type[ty]}_{i}'+F"\\{SSP[s]}_{Variable[v]}.nc" for i in Mode]
            # datasets=[xr.open_mfdataset(p,chunks={'time': -1,'lat':800, 'lon': 800}) for p in Files_name]
            datasets=[xr.open_mfdataset(p,chunks={'time': -1,'lat':800, 'lon': 800}) for p in Files_name]
            for i in range(len(datasets)):
                    datasets[i]['time'] = datasets[i]['time'].astype('datetime64[ns]')
                    datasets[i][Variable[v]] = (datasets[i][Variable[v]]/np.timedelta64(1,'D')).astype(int)
                    datasets[i][Variable[v]]=xr.where(datasets[i][Variable[v]].values<0,np.nan,datasets[i][Variable[v]])
                    datasets[i] = datasets[i].rename({Variable[v]:Variable[v]+str(i)})
            combined_datasets = xr.merge(datasets)
            CE_Frequency_Map = combined_datasets.to_array()
            print(output_path+Type[ty]+f'\\{Type[ty]}_{Variable[v]}_{SSP[s]}.nc')
            results=np.mean(CE_Frequency_Map,axis=0).rename(Variable[v])
            delayed_obj =results.to_netcdf(output_path+f'{Type[ty]}_{Variable[v]}_{SSP[s]}.nc', format='NETCDF4', engine='netcdf4',mode='w',compute=False)
            with ProgressBar():
                results222 = delayed_obj.compute()


