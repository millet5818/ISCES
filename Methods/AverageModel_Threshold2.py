import numpy as np
import xarray as xr
import pandas as pd
import os
from dask.diagnostics import ProgressBar

output_path = "E:\CE_DATA\Data_Processing\Average_Mode\Threshold\CMIP6\\"  # 输出结果所在文件夹
Mode=['BCC-CSM2-MR','CanESM5','CMCC-ESM2','GFDL-ESM4','IPSL-CM6A-LR','MIROC6','NESM3']
filePath='E:\CE_DATA\Data_Processing\Process_Results\Threshold\Tem\\'
Files_name=[filePath+F'{i}'+F"\\Historical_Threshold_Tem.nc" for i in Mode]
datasets=[xr.open_mfdataset(p) for p in Files_name]
for i in range(len(datasets)):
        datasets[i] = datasets[i].rename({'Threshold':'Threshold'+str(i)})
combined_datasets = xr.merge(datasets)
CE_Frequency_Map = combined_datasets.to_array()
results=np.mean(CE_Frequency_Map,axis=0).rename('Threshold')
delayed_obj =results.to_netcdf(output_path+'Tem_Threshold.nc', format='NETCDF4', engine='netcdf4',mode='w',compute=False)
with ProgressBar():
    results222 = delayed_obj.compute()


