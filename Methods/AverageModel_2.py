import numpy as np
import xarray as xr
import pandas as pd
import os
from dask.diagnostics import ProgressBar

output_path = "E:\CE_DATA\Data_Processing\Average_Mode\\"  # 输出结果所在文件夹
Type=['CE']
SSP=['Historical']
Mode=['ACCESS-CM2','BCC-CSM2-MR','CanESM5','CMCC-ESM2','GFDL-ESM4','IPSL-CM6A-LR','MIROC6','NESM3']
filePath='F:\Results\\'
# Variable=['Frequency','Amplitude','End','Start','Duration']
Variable=['Frequency']
for ty in range(len(Type)):
    for v in range(len(Variable)):
        for s in range(len(SSP)):
            Files_name=[filePath+F'{Type[ty]}_{i}'+F"\\CE_{Variable[v]}_"+SSP[s]+".nc" for i in Mode]
            datasets=[xr.open_mfdataset(p,chunks={'time': -1,'lat':800, 'lon': 800}) for p in Files_name]
            for i in range(len(datasets)):
                    datasets[i]['time'] = datasets[i]['time'].astype('datetime64[ns]')
                    datasets[i] = datasets[i].rename({'CE_'+Variable[v]: Mode[i]})
            combined_datasets = xr.merge(datasets)
            CE_Frequency_Map = combined_datasets.to_array()
            print(output_path+Type[ty]+f'\\{Type[ty]}_{Variable[v]}_{SSP[s]}.nc')
            results=np.mean(CE_Frequency_Map,axis=0).rename(Type[ty]+'_'+Variable[v])
            delayed_obj =results.to_netcdf(output_path+Type[ty]+f'\\{Type[ty]}_{Variable[v]}_{SSP[s]}.nc', format='NETCDF4', engine='netcdf4',mode='w',compute=False)
            with ProgressBar():
                results222 = delayed_obj.compute()


