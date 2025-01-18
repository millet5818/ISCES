import numpy as np
import xarray as xr
import pandas as pd
import os
from dask.diagnostics import ProgressBar
import xclim




output_path = "E:\CE_DATA\Data_Processing\Average_Mode\Daily_Pre\CMIP6\\"  # 输出结果所在文件夹
Mode=['ACCESS-CM2','BCC-CSM2-MR','CanESM5','CMCC-ESM2','GFDL-ESM4','IPSL-CM6A-LR','MIROC6','NESM3']
filePath='F:\Datasets\CMIP6\CMIP6_0.15_Binear\Pr\\'
filebase="F:\Datasets\CMIP6\CMIP6_0.15_Binear\Pr\ACCESS-CM2\Historical\pr_day_ACCESS-CM2_historical_r1i1p1f1_gn_2014.nc"

Times=np.arange(2014,2015,1)
for y in range(len(Times)):
    Files_name=[filePath+F'{i}'+F"\\Historical\\pr_day_{i}_historical_r1i1p1f1_gn_{Times[y]}.nc" for i in Mode]
    # Files_name=[filePath+F'{i}'+F"\\126\\pr_day_{i}_ssp126_r1i1p1f1_gn_{Times[y]}.nc" for i in Mode]
    print(Files_name)
    datasets=[xr.open_mfdataset(p,chunks={'time': -1,'lat':800, 'lon': 800})  for p in Files_name]
    for i in range(len(datasets)):
            datasets[i]['time'] = datasets[i]['time'].astype('datetime64[ns]')
            datasets[i]['pr'] = xclim.core.units.convert_units_to(datasets[i]['pr'], "mm/d")
            datasets[i] = datasets[i].rename({'pr': 'pr'+str(i)})
    combined_datasets = xr.merge(datasets)
    CE_Frequency_Map = combined_datasets.to_array()
    results=np.mean(CE_Frequency_Map,axis=0).rename('pr')
    delayed_obj =results.to_netcdf(output_path+f'Pr_{Times[y]}.nc', format='NETCDF4', engine='netcdf4',mode='w',compute=False)
    with ProgressBar():
        results222 = delayed_obj.compute()
