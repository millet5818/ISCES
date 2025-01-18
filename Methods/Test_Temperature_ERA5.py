import warnings

import numpy as np
from xclim.indices.generic import compare, select_resample_op, threshold_count
warnings.filterwarnings("ignore")
import xarray as xr
from xarray import open_mfdataset
from xclim.core.units import amount2rate
import xclim
from dask.diagnostics import ProgressBar

from xclim.indices import run_length as rl
from  xclim.core.units import to_agg_units

# TODO Number of wet days with daily precipitation over a given percentile
from xclim.indicators.atmos import days_over_precip_doy_thresh,days_over_precip_thresh,maximum_consecutive_wet_days
# TODO Fraction of precipitation due to wet days with daily precipitation over a given percentile:R95C,极端降雨对于降雨的贡献率
from xclim.indicators.atmos import fraction_over_precip_doy_thresh,wet_precip_accumulation,max_n_day_precipitation_amount
# TODO Fraction of precipitation over threshold during wet days
from xclim.indicators.atmos import fraction_over_precip_thresh,heat_wave_max_length
from xclim.indices import hot_spell_max_length
# todo Days with mean temperature above the 90th percentile
from xclim.indicators.atmos import tg90p,tg_days_above,tn90p,tn_days_above,tx90p,tx_days_above,tx_tn_days_above,warm_and_wet_days
from xclim.core.calendar import percentile_doy,resample_doy,get_calendar
import xarray
from xclim.core.utils import DayOfYearStr, Quantified, Quantity

# todo 以1mm为基准

def T955Max_L3( tasmax,tasmax_per):
    constrain = (">", ">=")
    thresh = resample_doy(tasmax_per, tasmax)
    cond = compare(tasmax, ">=", thresh, constrain)
    max_l = rl.resample_and_rl(
        cond,
        True,
        rl.longest_run,
        freq='YS',
    )
    # todo 这里只是把不想看到的隐藏了而已
    out = max_l.where(max_l >= 1, np.nan) #todo 1 指的是 window
    return to_agg_units(out, tasmax, "count")
# todo window参数指的是至少连续几天被筛选出来，这里设置为1指的是只要连续一天的都会被筛选出来，也就是至少连续一天的基础上的最大长度
def T955Max_L4(tasmax,tasmax_per):
    constrain = (">", ">=")
    thresh = resample_doy(tasmax_per, tasmax)
    cond = compare(tasmax, ">=", thresh, constrain)
    max_l = rl.resample_and_rl(
        cond,
        True,
        rl.rle_statistics,
        reducer='max',
        window=1,
        freq='YS',
    )
    # todo 这里只是把不想看到的隐藏了而已
    out = max_l.where(max_l >= 1, np.nan)
    return to_agg_units(out, tasmax, "count")
# todo 筛选事件数量，这里是指温度高于某个阈值的基础上，至少连续高于一天的事件数量
def T95Events(tasmax,tasmax_per):
    constrain = (">", ">=")
    thresh = resample_doy(tasmax_per, tasmax)
    cond = compare(tasmax, ">", thresh, constrain)
    max_l = rl.resample_and_rl(
        cond,
        True,
        rl.windowed_run_events,
        window=1,
        freq='YS',
    )
    # todo 这里只是把不想看到的隐藏了而已
    out = max_l.where(max_l >= 1, np.nan)
    return to_agg_units(out, tasmax, "count")
# todo 至少温度连续几天高于某个阈值的总长度（总天数）,这里暂时将window设置为1，也就等于寻找温度至少1天高于这个阈值的总长度，也是总天数
def T955Total_L(tasmax,tasmax_per):
    constrain = (">", ">=")
    thresh = resample_doy(tasmax_per, tasmax)
    cond = compare(tasmax, ">", thresh, constrain)
    max_l = rl.resample_and_rl(
        cond,
        True,
        rl.windowed_run_count,
        window=1,
        freq='YS',
    )
    # todo 这里只是把不想看到的隐藏了而已
    out = max_l.where(max_l >= 1, np.nan)
    return to_agg_units(out, tasmax, "count")
def T95Start_T(tasmax,tasmax_per):
    constrain = (">", ">=")
    thresh = resample_doy(tasmax_per, tasmax)
    cond = compare(tasmax, ">=", thresh, constrain)
    out = cond.resample(time='YS').map(
        rl.first_run,
        window=1,
        dim="time",
        coord="dayofyear",
    )
    out.attrs["units"] = ""
    return out
#todo 相比方法1，需要设置一个起始时间
def T95Start_T2(tasmax, tasmax_per):
    constrain = (">", ">=")
    thresh = resample_doy(tasmax_per, tasmax)
    cond = compare(tasmax, ">=", thresh, constrain)
    out: xarray.DataArray = cond.resample(time='YS').map(
        rl.first_run_after_date,
        window=1,
        date="01-01",
        dim="time",
        coord="dayofyear",
    )
    out.attrs.update(units="", is_dayofyear=np.int32(1), calendar=get_calendar(tasmax))
    return out

    # xclim.indices.generic.first_day_threshold_reached
    # xclim.indices.generic.first_occurrence
    # xclim.indices.generic.last_occurrence
def T95End_T(tasmax, tasmax_per):
    constrain = (">", ">=")
    thresh = resample_doy(tasmax_per, tasmax)
    cond = compare(tasmax, ">=", thresh, constrain)
    out: xarray.DataArray = cond.resample(time='YS').map(
        rl.last_run,
        window=1,
        dim="time",
        coord="dayofyear",
    )
    out.attrs["units"] = ""
    return out

def Tem_Related2():
    print("与温度相关")
    """definition1:daily tem>32℃ and >( prethsold:0.7,0.8,0.9 based 15 day-time windows"""
    # cond = (compare(data['mx2t'], ">", T90, constrain)) & (
    #     compare(data['mx2t'], ">", T90, constrain)
    # )


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
    r75p.rename('R75P').to_netcdf("F:\Experiments\Data_Processing\Results\ERA5_Tem", format='NETCDF4', engine='netcdf4')

    R95C=r75p/(wetdays_Array.groupby(r75p_year.time.dt.year).sum(dim='time'))

    r95p_year=xr.where(data.tp>p75,data.tp,0)
    r95p=r95p_year.groupby(r95p_year.time.dt.year).sum(dim='time')

    r95i=r95p/r75p # 降雨强度


Time_Base=np.arange(1950,2015,1)
filename="E:\CE_DATA\Data_Processing\ERA5\Max_Tem\\*.nc"

print("与温度相关")
"""definition1:tem >prethsold:0.7,0.8,0.9"""
data = open_mfdataset(filename,parallel=True, concat_dim="time",chunks={'time': -1, 'latitude': 100, 'longitude': 100}, combine="nested",data_vars='minimal', coords='minimal', compat='override')
data['mx2t'] = xclim.core.units.convert_units_to(data['mx2t'], "degC")
Files_Base=data.where(data.time.dt.year.isin(Time_Base),drop=True)

# T90 = percentile_doy(Files_Base['mx2t'], per=[70], window=15).sel(percentiles=[70])
T90 = percentile_doy(Files_Base['mx2t'], per=[70,80,90], window=15).sel(percentiles=[70,80,90])

# TODO Percentile
Percentile_P=T90.rename('Threshold').to_netcdf("E:\CE_DATA\Data_Processing\Process_Results\\Historical_Threshold.nc", format='NETCDF4', engine='netcdf4', compute=False)
with ProgressBar():
    Percentile_P.compute()

# TODO T90D
T90D = T955Total_L(data['mx2t'], T90) #tx90p(data['mx2t'], T90),该方法也可以
T90D_P = T90D.rename('T90D').to_netcdf("E:\CE_DATA\Data_Processing\Process_Results\\T95D.nc", format='NETCDF4', engine='netcdf4', compute=False)
with ProgressBar():
    T90D_P.compute()

# TODO T95Max_L
T95Max_L=T955Max_L4(data['mx2t'],T90) #T955Max_L3(data['mx2t'],T90)该方法也可以
T95Max_L_P = T95Max_L.rename('T95Max_L').to_netcdf("E:\CE_DATA\Data_Processing\Process_Results\\T95Max_L.nc", format='NETCDF4', engine='netcdf4', compute=False)
with ProgressBar():
    T95Max_L_P.compute()

# TODO T95Events
T95Events_L=T95Events(data['mx2t'],T90)
T95Events_L_P = T95Events_L.rename('T95Events').to_netcdf("E:\CE_DATA\Data_Processing\Process_Results\\T95Events.nc", format='NETCDF4', engine='netcdf4', compute=False)
with ProgressBar():
    T95Events_L_P.compute()

#TODO T95Start_T
T95Start_T_P=T95Start_T2(data['mx2t'],T90) # T95Start_T也可以实现
T95Start_T_P_L = T95Start_T_P.rename('T95Start_T').to_netcdf("E:\CE_DATA\Data_Processing\Process_Results\\T95Start_T.nc", format='NETCDF4', engine='netcdf4', compute=False)
with ProgressBar():
    T95Start_T_P_L.compute()

# TODO T95End_T
T95End_T_P=T95End_T(data['mx2t'],T90) # T95Start_T也可以实现
T95End_T_P_L = T95End_T_P.rename('T95End_T').to_netcdf("E:\CE_DATA\Data_Processing\Process_Results\\T95End_T.nc", format='NETCDF4', engine='netcdf4', compute=False)
with ProgressBar():
    T95End_T_P_L.compute()





# 需要计算的指数有

# TODO T95D, T95Max_L, T95Start_T, T95End_T

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
