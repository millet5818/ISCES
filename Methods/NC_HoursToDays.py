"""
转换ERA5 MC 下载的是逐小时数据，需要转换为逐日数据单个文件
"""
import xarray as xr
import numpy as np
import pandas as pd
def HourToDay(filePath):
    for y in range(1979,2023):
        y_array_file=[]
        for m in range(1,13):
            fileName=filePath+"total_precipitation_"+str(y)+str(m).zfill(2)+".nc"
            ds = xr.open_dataset(fileName)
            daily_total_precip = ds.tp.resample(time='1D').sum(dim='time')
            #daily_avg_temp = ds.t2m.resample(time='1D').mean(dim='time')
            y_array_file.append(daily_total_precip)
            del daily_total_precip
        Y_NC=xr.concat(y_array_file,dim='time')
        Y_NC.to_netcdf(filePath+"total_precipitation_"+str(y)+'.nc')
        del y_array_file

# HourToDay('G:\\Datasets\\Climate Data\\remote sensing product_Reanalysis product\ERA5\\total_precipitation\\')
da = xr.DataArray([[2,3,4],[5,3,2],[2,np.nan,232]])
print(da)