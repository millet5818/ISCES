import warnings
warnings.filterwarnings("ignore")
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
import xclim.indicators
from xclim import ensembles
import xclim.indices as xi
import xclim.core.units
import dask
from xarray import open_mfdataset
from xclim import testing
from glob import glob
from xclim.core.calendar import percentile_doy,resample_doy
from xclim.indices.generic import threshold_count,compare
from xclim.ensembles import create_ensemble,ensemble_percentiles
from xclim.indices import days_over_precip_thresh
from dask.distributed import Client
from dask.diagnostics import ProgressBar

# TODO Number of days with temperature below a given percentile and precipitation above a given percentile.
from  xclim.indicators.atmos import cold_and_wet_days
# TODO Number of wet days with daily precipitation over a given percentile
from xclim.indicators.atmos import days_over_precip_doy_thresh,days_over_precip_thresh,maximum_consecutive_wet_days
# TODO Fraction of precipitation due to wet days with daily precipitation over a given percentile:R95C,极端降雨对于降雨的贡献率
from xclim.indicators.atmos import fraction_over_precip_doy_thresh,wet_precip_accumulation,max_n_day_precipitation_amount
# TODO Fraction of precipitation over threshold during wet days
from xclim.indicators.atmos import fraction_over_precip_thresh

# if __name__ == '__main__':
#     # client=Client(memory_limit='20GB',n_workers=1)
#     # client=Client(processes=False,n_workers=1,threads_per_worker=10, memory_limit='15GB')# TODO 不出现任何消息输出的方式 or 把下面的代码放在一起
#     client=Client(n_workers=1,threads_per_worker=20, memory_limit='20GB')

# todo 以1mm为基准
Time_Base=np.arange(1950,2015,1)
FileNameList="F:\Datasets\CMIP6\CMIP6_0.15_Binear\Pr\MIROC6\\Historical\\*.nc"
FileNameList2="F:\Datasets\CMIP6\CMIP6_0.15_Binear\Pr\MIROC6\\126\\*.nc"
FileNameList3="F:\Datasets\CMIP6\CMIP6_0.15_Binear\Pr\MIROC6\\245\\*.nc"
FileNameList4="F:\Datasets\CMIP6\CMIP6_0.15_Binear\Pr\MIROC6\\585\\*.nc"


HistoricalFile=open_mfdataset(FileNameList,parallel=True,concat_dim="time",chunks={'time': -1,'lat':200, 'lon': 200}, combine="nested",data_vars='minimal', coords='minimal', compat='override')
HistoricalFile['pr'] = xclim.core.units.convert_units_to(HistoricalFile['pr'], "mm/d")
# todo 计算历史时期的阈值
Files_Base=HistoricalFile.where(HistoricalFile.time.dt.year.isin(Time_Base),drop=True)
R_threshold = Files_Base.pr.where(Files_Base.pr >= 1)
# Threshold_Base = R_threshold.chunk({"time": len(R_threshold.time), "lat": 400, "lon": 400}).quantile([0.7,0.8,0.9], dim="time", keep_attrs=True)
Threshold_Base = R_threshold.chunk({"time": -1, "lat": 200, "lon": 200}).quantile([0.7,0.8,0.9], dim="time", keep_attrs=True)
aaa=Threshold_Base.rename('Threshold').to_netcdf("E:\\CE_DATA\\Data_Processing\\Process_Results\\MIROC6\\Historical_Threshold.nc",format='NETCDF4', engine='netcdf4',compute=False)
with ProgressBar():
    aaa.compute()

#TODO 这里读取这个值，减少后面的内存需求
Threshold_Base=xr.open_dataset("E:\\CE_DATA\\Data_Processing\\Process_Results\\MIROC6\\Historical_Threshold.nc")
# #
# # # todo 计算历史时期的R95D
# R95D = days_over_precip_thresh(Files_Base.pr, Threshold_Base.Threshold)
# bbb=R95D.rename('R95D').to_netcdf(f"E:\\CE_DATA\\Data_Processing\\Process_Results\\MIROC6\\Historical_R95D.nc", format='NETCDF4', engine='netcdf4',compute=False)
# with ProgressBar():
#     results = bbb.compute()
#
# # todo 计算历史时期的R95C
# R95C= fraction_over_precip_thresh(Files_Base.pr, Threshold_Base.Threshold)
# cc=R95C.rename('R95C').to_netcdf("E:\\CE_DATA\\Data_Processing\\Process_Results\\MIROC6\\Historical_R95C.nc", format='NETCDF4', engine='netcdf4',compute=False)
# with ProgressBar():
#        cc.compute()
# # todo 计算历史时期的R95P
# R95P=wet_precip_accumulation(pr=Files_Base.pr,thresh=Threshold_Base.Threshold)
# ddd=R95P.rename('R95P').to_netcdf("E:\\CE_DATA\\Data_Processing\\Process_Results\\MIROC6\\Historical_R95P.nc",format='NETCDF4', engine='netcdf4',compute=False)
# with ProgressBar():
#      ddd.compute()
# # todo 计算历史时期的R95DM
# R95DM=maximum_consecutive_wet_days(pr=Files_Base.pr,thresh=Threshold_Base.Threshold)
# eee=R95DM.rename('R95DM').to_netcdf("E:\\CE_DATA\\Data_Processing\\Process_Results\\MIROC6\\Historical_R95DM.nc",format='NETCDF4', engine='netcdf4',compute=False)
# with ProgressBar():
#       eee.compute()
#
#
# # TODO 126的计算
# Files_126 =open_mfdataset(FileNameList2,parallel=True,concat_dim="time",chunks={'time': -1,'lat':200, 'lon': 200}, combine="nested",data_vars='minimal', coords='minimal', compat='override')
# Files_126['pr'] = xclim.core.units.convert_units_to(Files_126['pr'], "mm/d")
# # todo 计算126时期的R95D
# R95D_126 = days_over_precip_thresh(Files_126.pr, Threshold_Base.Threshold)
# bbb_126=R95D_126.rename('R95D').to_netcdf("E:\\CE_DATA\\Data_Processing\\Process_Results\\MIROC6\\126_R95D.nc", format='NETCDF4', engine='netcdf4',compute=False)
# with ProgressBar():
#      bbb_126.compute()
#
# # todo 计算126时期的R95C
# R95C_126= fraction_over_precip_thresh(Files_126.pr, Threshold_Base.Threshold)
# cc_126=R95C_126.rename('R95C').to_netcdf("E:\\CE_DATA\\Data_Processing\\Process_Results\\MIROC6\\126_R95C.nc", format='NETCDF4', engine='netcdf4',compute=False)
# with ProgressBar():
#        cc_126.compute()
#
# # todo 计算126时期的R95P
# R95P_126=wet_precip_accumulation(pr=Files_126.pr,thresh=Threshold_Base.Threshold)
# ddd_126=R95P_126.rename('R95P').to_netcdf("E:\\CE_DATA\\Data_Processing\\Process_Results\\MIROC6\\126_R95P.nc",format='NETCDF4', engine='netcdf4',compute=False)
# with ProgressBar():
#      ddd_126.compute()
# # todo 计算126时期的R95DM
# R95DM_126=maximum_consecutive_wet_days(pr=Files_126.pr,thresh=Threshold_Base.Threshold)
# eee_126=R95DM_126.rename('R95DM').to_netcdf("E:\\CE_DATA\\Data_Processing\\Process_Results\\MIROC6\\126_R95DM.nc",format='NETCDF4', engine='netcdf4',compute=False)
# with ProgressBar():
#       eee_126.compute()
#
# #
# # TODO 245的计算
# Files_245 =open_mfdataset(FileNameList3,parallel=True,concat_dim="time",chunks={'time': -1,'lat':200, 'lon': 200}, combine="nested",data_vars='minimal', coords='minimal', compat='override')
# Files_245['pr'] = xclim.core.units.convert_units_to(Files_245['pr'], "mm/d")
# # todo 计算245时期的R95D
# R95D_245 = days_over_precip_thresh(Files_245.pr, Threshold_Base.Threshold)
# bbb_245=R95D_245.rename('R95D').to_netcdf("E:\\CE_DATA\\Data_Processing\\Process_Results\\MIROC6\\245_R95D.nc", format='NETCDF4', engine='netcdf4',compute=False)
# with ProgressBar():
#      bbb_245.compute()
# # todo 计算245时期的R95C
# R95C_245= fraction_over_precip_thresh(Files_245.pr, Threshold_Base.Threshold)
# cc_245=R95C_245.rename('R95C').to_netcdf("E:\\CE_DATA\\Data_Processing\\Process_Results\\MIROC6\\245_R95C.nc", format='NETCDF4', engine='netcdf4',compute=False)
# with ProgressBar():
#        cc_245.compute()
# # todo 计算245时期的R95P
# R95P_245=wet_precip_accumulation(pr=Files_245.pr,thresh=Threshold_Base.Threshold)
# ddd_245=R95P_245.rename('R95P').to_netcdf("E:\\CE_DATA\\Data_Processing\\Process_Results\\MIROC6\\245_R95P.nc",format='NETCDF4', engine='netcdf4',compute=False)
# with ProgressBar():
#      ddd_245.compute()
# # todo 计算245时期的R95DM
# R95DM_245=maximum_consecutive_wet_days(pr=Files_245.pr,thresh=Threshold_Base.Threshold)
# eee_245=R95DM_245.rename('R95DM').to_netcdf("E:\\CE_DATA\\Data_Processing\\Process_Results\\MIROC6\\245_R95DM.nc",format='NETCDF4', engine='netcdf4',compute=False)
# with ProgressBar():
#       eee_245.compute()
# #
# TODO 585的计算
Files_585 =open_mfdataset(FileNameList4,parallel=True,concat_dim="time",chunks={'time': -1,'lat':200, 'lon':200}, combine="nested",data_vars='minimal', coords='minimal', compat='override')
Files_585['pr'] = xclim.core.units.convert_units_to(Files_585['pr'], "mm/d")
# # todo 计算585时期的R95D
# R95D_585 = days_over_precip_thresh(Files_585.pr, Threshold_Base.Threshold)
# bbb_585=R95D_585.rename('R95D').to_netcdf("E:\\CE_DATA\\Data_Processing\\Process_Results\\MIROC6\\585_R95D.nc", format='NETCDF4', engine='netcdf4',compute=False)
# with ProgressBar():
#      bbb_585.compute()
# todo 计算585时期的R95C
R95C_585= fraction_over_precip_thresh(Files_585.pr, Threshold_Base.Threshold)
cc_585=R95C_585.rename('R95C').to_netcdf("E:\\CE_DATA\\Data_Processing\\Process_Results\\MIROC6\\585_R95C.nc", format='NETCDF4', engine='netcdf4',compute=False)
with ProgressBar():
       cc_585.compute()
# todo 计算585时期的R95P
R95P_585=wet_precip_accumulation(pr=Files_585.pr,thresh=Threshold_Base.Threshold)
ddd_585=R95P_585.rename('R95P').to_netcdf("E:\\CE_DATA\\Data_Processing\\Process_Results\\MIROC6\\585_R95P.nc",format='NETCDF4', engine='netcdf4',compute=False)
with ProgressBar():
     ddd_585.compute()
# todo 计算585时期的R95DM
R95DM_585=maximum_consecutive_wet_days(pr=Files_585.pr,thresh=Threshold_Base.Threshold)
eee_585=R95DM_585.rename('R95DM').to_netcdf("E:\\CE_DATA\\Data_Processing\\Process_Results\\MIROC6\\585_R95DM.nc",format='NETCDF4', engine='netcdf4',compute=False)
with ProgressBar():
      eee_585.compute()




