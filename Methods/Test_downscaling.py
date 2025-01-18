import xarray as xr
import numpy as np
import xclim.core.units
import xclim
from xclim import sdba
import dask
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from xclim import ensembles
from pathlib import Path
import glob
from dask.diagnostics import ProgressBar
import warnings
from dask.distributed import Client

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    # client=Client(memory_limit='20GB',n_workers=1)
    # client=Client(processes=False,n_workers=1,threads_per_worker=10, memory_limit='15GB')# TODO 不出现任何消息输出的方式 or 把下面的代码放在一起
    client=Client(n_workers=1,threads_per_worker=10, memory_limit='10GB')


Obs_filename="E:\CE_DATA\Data_Processing\ERA5\\Pr\\"
Downscaling_filename="E:\CE_DATA\Data_Processing\CMIP6\\Pr\\ACCESS-CM2\\126\\"
Historical_filename="E:\CE_DATA\Data_Processing\CMIP6\Pr\\ACCESS-CM2\\Historical\\"
outputFileName="E:\CE_DATA\Data_Processing\Process_Results\\new_file.nc"
outputFileNam1e="E:\CE_DATA\Data_Processing\Process\\delat_file.nc"
# 选择时间范围
Time_range=np.arange(1950,2014,1)
with xr.open_mfdataset(Obs_filename+"*.nc",concat_dim="time", combine="nested",data_vars='minimal',chunks={'time': -1,'latitude': 5, 'longitude': 5}, coords='minimal', compat='override') as Obs_file:
    Obs_file_Filter=Obs_file.where((Obs_file.time.dt.year.isin(Time_range)), drop=True)
with xr.open_mfdataset(Historical_filename+"*.nc",concat_dim="time", combine="nested",data_vars='minimal',chunks={'time': -1,'latitude': 5, 'longitude': 5}, coords='minimal', compat='override') as Historical_file:
    Historical_file_Filter=Historical_file.where((Historical_file.time.dt.year.isin(Time_range)), drop=True)
# with xr.open_mfdataset(Downscaling_filename+"*.nc",concat_dim="time", combine="nested",data_vars='minimal', chunks={'time': -1,'latitude': 50, 'longitude': 50},coords='minimal', compat='override') as Dowscaling_file:
#     Downscaling_file_Filter=Dowscaling_file

# 计算多年日平均
# 将单位均转换为mm/day,这里涉及降雨的
if Obs_file_Filter['tp'].attrs["units"]!='mm':
   # Obs_file_Filter=Obs_file_Filter['tp']*1000
   # Obs_file_Filter['tp'].attrs["units"]='mm/d'
   Obs_file_Filter['tp']=xclim.core.units.convert_units_to(Obs_file_Filter['tp'],"mm")
   Historical_file_Filter['pr']=xclim.core.units.convert_units_to(Historical_file_Filter['pr'],"mm")
   # Downscaling_file_Filter['pr']=xclim.core.units.convert_units_to(Downscaling_file_Filter['pr'],"mm")

# drop-移除Coordinates/重命名
Historical_file_Filter = Historical_file_Filter.rename({'lat':'latitude','lon':'longitude'})
# Downscaling_file_Filter = Downscaling_file_Filter.rename({'lat':'latitude','lon':'longitude'})

QM = sdba.EmpiricalQuantileMapping.train(Obs_file_Filter.tp.chunk({"time": -1, "latitude": 5, "longitude": 5}), Historical_file_Filter.pr.chunk({"time": -1, "latitude": 5, "longitude": 5}), nquantiles=15, group="time", kind="+")
scen = QM.adjust(Historical_file_Filter.pr.chunk({"time": -1, "latitude": 5, "longitude": 5}), extrapolation="constant", interp="nearest")
delayed_obj=scen.to_netcdf(outputFileName, format='NETCDF4', engine='netcdf4',mode='w',compute=False)
with ProgressBar():
    results = delayed_obj.compute()


#
# # 排序
# Obs_file_Filter_multi_Day_Mean = Obs_file_Filter.groupby(Obs_file_Filter.time.dt.dayofyear).mean()
# Historical_file_Filter_multi_Day_Mean = Historical_file_Filter.groupby(Historical_file_Filter.time.dt.dayofyear).mean()
# # 计算偏差
# Historical_file_Filter_multi_Day_Mean_Fill0=np.where(Historical_file_Filter_multi_Day_Mean.pr==0,Historical_file_Filter_multi_Day_Mean.pr,Obs_file_Filter_multi_Day_Mean.tp / Historical_file_Filter_multi_Day_Mean.pr)
# # Historical_file_Filter_multi_Day_Mean_Fill0=np.where(Historical_file_Filter_multi_Day_Mean.pr==0,np.nan,Historical_file_Filter_multi_Day_Mean.pr)
# # Historical_file_Filter_multi_Day_Mean_Fill0_nc = xr.Dataset({"pr":
# #                          (('dayofyear', 'latitude', 'longitude'), Historical_file_Filter_multi_Day_Mean_Fill0)},
# #                          coords={"time": Historical_file_Filter_multi_Day_Mean.dayofyear.values,
# #                             "latitude": Historical_file_Filter_multi_Day_Mean.latitude.values,
# #                             "longitude": Historical_file_Filter_multi_Day_Mean.longitude.values})
# #
# # delta_pre = Obs_file_Filter_multi_Day_Mean.tp / Historical_file_Filter_multi_Day_Mean_Fill0_nc.pr
# # Historical_file_Filter_multi_Day_Mean_Fill_nan=np.where(Historical_file_Filter_multi_Day_Mean.pr==0,Historical_file_Filter_multi_Day_Mean.pr,delta_pre.pr)
# Historical_file_Filter_multi_Day_Mean_Fill0_nc = xr.Dataset({"pr":
#                          (('dayofyear', 'latitude', 'longitude'), Historical_file_Filter_multi_Day_Mean_Fill0)},
#                          coords={"time": Historical_file_Filter_multi_Day_Mean.dayofyear.values,
#                             "latitude": Historical_file_Filter_multi_Day_Mean.latitude.values,
#                             "longitude": Historical_file_Filter_multi_Day_Mean.longitude.values})
#
# Historical_file_Filter_multi_Day_Mean_Fill0_nc.to_netcdf(outputFileNam1e, format='NETCDF4', engine='netcdf4')
# # downcaling
# result = []
# for i in range(1,367):
#      itmp = Historical_file_Filter.sel(time=Historical_file_Filter.time.dt.dayofyear == i)
#      # 由于delta_pre的时间维度是整数，直接用月份索引来获取对应的偏差值
#      delta_pre_day = Historical_file_Filter_multi_Day_Mean_Fill0_nc.sel(dayofyear=i)
#      itmp_adjusted = itmp.copy()
#      itmp_adjusted['pr'] = itmp.pr * delta_pre_day
#      # 添加处理后的数据到结果列表
#      result.append(itmp_adjusted)
# # 合并结果列表中的所有数据集
# pre_gcm_downscaled = xr.concat(result, dim='time')
# pre_gcm_downscaled.to_netcdf(outputFileName, format='NETCDF4', engine='netcdf4')
#
# # enc = {}
# # for k in pre_gcm_downscaled.data_vars:
# #       if pre_gcm_downscaled[k].ndim < 2:
# #         continue
# #       enc[k] = {
# #                 "zlib": True,
# #                 "complevel": 3,
# #                 "fletcher32": True,
# #                 # "chunksizes": tuple(map(lambda x: x//2, u_downsampled[k].shape))
# #                 # "chunksizes": (1,600,1440)
# #         }
#
# # 计算精度指标
# # 由于数据可能包含多个维度（例如，lat和lon），需要将数据平坦化以进行一对一比较
# # downscaled_values = pre_gcm_downscaled.drop('dayofyear').pr.values.ravel()
# # era5_values = Obs_file_Filter.tp.values.ravel()
# # pearson_corr, _ = pearsonr(downscaled_values, era5_values)
# # # 计算RMSE
# # rmse = np.sqrt(mean_squared_error(era5_values, downscaled_values))
# # print(111)
#
# # kg m-2 s-1
# # mm
# # mm/d
# # m
