import warnings

import numpy as np
from xclim.indices.generic import compare, select_resample_op, threshold_count
warnings.filterwarnings("ignore")
import xarray as xr
from xarray import open_mfdataset
from xclim.core.units import amount2rate
import xclim
from dask.diagnostics import ProgressBar
from timeit import default_timer


# TODO Number of days with temperature below a given percentile and precipitation above a given percentile.
from  xclim.indicators.atmos import cold_and_wet_days
# TODO Number of wet days with daily precipitation over a given percentile
from xclim.indicators.atmos import days_over_precip_doy_thresh,days_over_precip_thresh,maximum_consecutive_wet_days
# TODO Fraction of precipitation due to wet days with daily precipitation over a given percentile:R95C,极端降雨对于降雨的贡献率
from xclim.indicators.atmos import fraction_over_precip_doy_thresh,wet_precip_accumulation,max_n_day_precipitation_amount
# TODO Fraction of precipitation over threshold during wet days
from xclim.indicators.atmos import fraction_over_precip_thresh

# todo Days with mean temperature above the 90th percentile
from xclim.indicators.atmos import tg90p,tg_days_above,tn90p,tn_days_above,tx90p,tx_days_above,tx_tn_days_above,warm_and_wet_days
from xclim.core.calendar import percentile_doy
import dask
import dask.config
# 设置Dask的调度器为进程池，并指定工作进程数
# dask.config.set(scheduler='processes', num_workers=4)


filename=["E:\CE_DATA\Original_Data\ERA5\Max_2mTem\Maximum 2m temperature since previous post-processing_1950.nc",
          "E:\CE_DATA\Original_Data\ERA5\Max_2mTem\Maximum 2m temperature since previous post-processing_1951.nc"]

# filename=["E:\CE_DATA\Data_Processing\CMIP6\Pr\ACCESS-CM2\Historical\pr_day_ACCESS-CM2_historical_r1i1p1f1_gn_1950.nc",
#           "E:\CE_DATA\Data_Processing\CMIP6\Pr\ACCESS-CM2\Historical\pr_day_ACCESS-CM2_historical_r1i1p1f1_gn_1951.nc"]
outputfile="E:\CE_DATA\Data_Processing\Process_Results\\R95P3_test5.nc"


def Pre_Related():
    Threshold=[0.7,0.8,0.9]
    # data1=xr.open_dataset(filename[0])
    # data2=xr.open_dataset(filename[1])
    # data=xr.concat([data1,data2],dim='time')
    data=open_mfdataset(filename,parallel=True,concat_dim="time", chunks={'time': -1,'latitude': 100, 'longitude': 100},combine="nested")
    # p75 = data.tp.chunk({"time": len(data.time), "latitude": 100, "longitude": 100}).quantile(0.75, dim="time", keep_attrs=True)
    if (data['tp'].attrs["units"]!="kg m-2 s-1"):
        data['tp'] =xclim.core.units.amount2rate(data['tp'], out_units="mm/d")
    else:
        data['tp'] = xclim.core.units.convert_units_to(data['tp'], out_units="mm/d")
    # data['tp'] = xclim.core.units.amount2rate(data['tp'], out_units="mm/d")
    # data['tp']= xclim.core.units.convert_units_to(data['tp'], "mm/d")#todo 这个更高端
    R_threshold = data['tp'].where(data['tp'] >= 1) # todo 以 wetday 为基准
    for i in Threshold:
        p75 = R_threshold.chunk({"time": len(R_threshold.time), "latitude": 100, "longitude": 100}).quantile(i, dim="time", keep_attrs=True)
        r75p = days_over_precip_thresh(data.tp, p75)
        R95C= fraction_over_precip_thresh(data.tp, p75)
        R95P=wet_precip_accumulation(pr=data.tp,thresh=p75)
        R95DM=maximum_consecutive_wet_days(pr=data.tp,thresh=p75)
        # r75p_year = compare(data.tp, ">", p75, ">") * 1
        # r75p_year.attrs["units"] = ""
        # r75p_year.rename('Binary_Map').to_netcdf("E:\CE_DATA\Data_Processing\Process_Results\\Binary_Map.nc")
        p75.rename('Threshold').to_netcdf(F"E:\CE_DATA\Data_Processing\Process_Results\\Threshold_{i}.nc")
        r75p.to_netcdf(F"E:\CE_DATA\Data_Processing\Process_Results\\R95D_{i}.nc", format='NETCDF4', engine='netcdf4')
        R95C.to_netcdf(F"E:\CE_DATA\Data_Processing\Process_Results\\R95C_{i}.nc", format='NETCDF4', engine='netcdf4')
        R95P.to_netcdf(F"E:\CE_DATA\Data_Processing\Process_Results\\R95P_{i}.nc", format='NETCDF4', engine='netcdf4')
        R95DM.to_netcdf(F"E:\CE_DATA\Data_Processing\Process_Results\\R95DM_{i}.nc", format='NETCDF4', engine='netcdf4')
def R95D2():
    data1=xr.open_dataset(filename[0])
    data2=xr.open_dataset(filename[1])
    data=xr.concat([data1,data2],dim='time')
    data['tp'] = xclim.core.units.amount2rate(data['tp'], out_units="mm/d")
    wetdays_Array = data['tp'].where(data['tp'] >= 1)
    p75 = wetdays_Array.quantile(0.75, dim="time", keep_attrs=True)
    p75.rename('Threshold').to_netcdf("E:\CE_DATA\Data_Processing\Process_Results\\Threshold.nc")
    # TODO Binary Map
    # r75p_year=xr.where(data.tp>p75,1,0) ===等价于compare(data.tp, ">", p75, ">") * 1
    r75p_year = compare(data.tp, ">", p75, ">") * 1
    r75p_year.attrs["units"] = ""
    r75p_year.rename('Binary_Map').to_netcdf("E:\CE_DATA\Data_Processing\Process_Results\\Binary_Map.nc")
    r75p=r75p_year.groupby(r75p_year.time.dt.year).sum(dim='time')
    r75p.attrs["units"] = "d"
    r75p.rename('R75P').to_netcdf(outputfile, format='NETCDF4', engine='netcdf4')

    R95C=r75p/(wetdays_Array.groupby(r75p_year.time.dt.year).sum(dim='time'))

    r95p_year=xr.where(data.tp>p75,data.tp,0)
    r95p=r95p_year.groupby(r95p_year.time.dt.year).sum(dim='time')

    r95i=r95p/r75p # 降雨强度
def Tem_Related():
    print("与温度相关")
    """definition1:tem >prethsold:0.7,0.8,0.9"""
def Tem_Related2():
    print("与温度相关")
    """definition1:daily tem>32℃ and >( prethsold:0.7,0.8,0.9 based 15 day-time windows"""
#TODO 可以不for循环，但我的电脑内存太小，当运行R795P时，会炸裂，故分开运行
data1=xr.open_dataset(filename[0])
data2=xr.open_dataset(filename[1])
data=xr.concat([data1,data2],dim='time')
# data['tp']= xclim.core.units.amount2rate(data['tp'], out_units="mm/d")
# data['tp']= xclim.core.units.convert_units_to(data['tp'], "mm/d")
# obs_pre_monthlymean = obs_pre_ds_re2.tp * 1000
# gcm_pre_monthlymean = gcm_pre_ds_re.pr * 60 * 60 *24
# data['pr']=data['pr']* 60 * 60 *24
data['mx2t'] = xclim.core.units.convert_units_to(data['mx2t'], "degC")
# R_threshold = data.tp.where(data.tp >= 1) # todo 以 wetday 为基准
p75 = data['mx2t'].percentile_doy(0.75).sel(percentiles=90)
r75p = tx90p(data.mx2t, p75)
sdss=r75p.to_netcdf(outputfile, format='NETCDF4', engine='netcdf4',compute=False)
with ProgressBar():
    sdss.compute()
# degree_days_exceedance_date
# TODO Number of days that constitute heatwave events
# heat_wave_index
# heat_wave_total_length
# heat_wave_max_length
# high_precip_low_temp()
# hot_spell_frequency
# hot_spell_max_length
# hot_spell_total_length
# ice_days
# frost_days
# maximum_consecutive_tx_days
# maximum_consecutive_wet_days
# wetdays_prop
# wetdays
# maximum_consecutive_tx_days
# heat_wave_frequency
# heat_wave_total_length
# maximum_consecutive_wet_days
# warm_day_frequency


# todo method record
# wetdays_Array = data['tp'].where(data['tp'] >= 1) 符合条件的保留，不符合的会变成NAN
# wetdays_Array = xr.where(data['tp'] >= 1,1,0) 符合条件的保留，不符合的会变成NAN
# todo 修改参数
# r75p_year.name="Binary_Map" === r75p_year.rename("Binary_Map")
# r75p_year.attrs["units"] = ""
# r75p_year.drop('quantile')


# data['tp']= xclim.core.units.amount2rate(data['tp'], out_units="mm/d")
# data['tp']= xclim.core.units.convert_units_to(data['tp'], "mm/d")
# data['pr']=data['pr']* 60 * 60 *24
# data['tp']=data['tp']* 1000
# data['tp'].attrs["units"]="mm/d"
