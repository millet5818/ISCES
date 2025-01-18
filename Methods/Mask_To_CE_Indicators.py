"""
Test CE
"""
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

from  xclim.indicators.atmos import warm_and_wet_days
from xclim.core.units import amount2rate
import xclim
from dask.diagnostics import ProgressBar
import os
import glob
from scipy.signal import convolve,choose_conv_method,convolve2d
from xclim.indices import run_length as rl
from  xclim.core.units import to_agg_units
import xarray
from functools import partial



# TODO  CE_Duration复合事件开始到结束的持续总天数，（Duration）,称为，复合事件发生的总天数
def CE_Duration(cond):
    max_l = rl.resample_and_rl(
        cond,
        True,
        rl.windowed_run_count,
        window=1,
        freq='YS',
    )
    # todo 这里只是把不想看到的隐藏了而已
    out = max_l.where(max_l >= 1, np.nan)
    out.attrs["units"] = 'd'
    return out
# TODO  CE_Frequency 复合事件开始到结束的持续总发生数量，（Frequency），称为复合事件一年发生的次数
def CE_Frequency(cond):
    max_l = rl.resample_and_rl(
        cond,
        True,
        rl.windowed_run_events,
        window=1,
        freq='YS',
    )
    # todo 这里只是把不想看到的隐藏了而已
    out = max_l.where(max_l >= 1, np.nan)
    out.attrs["units"] = ''
    return out
# TODO CE_Amplitude 复合事件中发生持续天数最长的天数（amplitude）称为波幅
def CE_Amplitude(cond):
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
    out.attrs["units"] = 'd'
    return out

# TODO CE_Return 复合事件的重放周期(返回期)
def CE_Return(cond):
    Pr_Tem_YS=cond.sum(dim="time")
    Return_Period=Pr_Tem_YS/cond.time.dt.day.shape[0]
    Return_Period=1/(Return_Period*365)
    Return_Period = Return_Period.where(~np.isinf(Return_Period))
    # Return_Period[np.isinf(Return_Period)] = np.nan
    Return_Period.attrs["units"] = "yrs"
    return Return_Period

# TODO CE_Ratio_PR 强降雨对复合事件的贡献率
def CE_Ratio_PR(cond,Pr):
    over = (
        cond
        .resample(time='YS')
        .sum(dim="time")
    )

    # Compute the days when precip is both over the wet day threshold and the percentile threshold.
    total = (
        Pr.resample(time='YS').sum(dim="time")
    )

    out = over / total
    out.attrs["units"] = ""
    return out

# TODO CE_Ratio_TEM 高温对复合事件的贡献率
def CE_Ratio_Tem(cond,Tem):
    over = (
        cond
        .resample(time='YS')
        .sum(dim="time")
    )

    # Compute the days when precip is both over the wet day threshold and the percentile threshold.
    total = (
        Tem.resample(time='YS').sum(dim="time")
    )

    out = over / total
    out.attrs["units"] = ""
    return out

# TODO CE_Start 第一起复合事件的开始时间
def CE_Start(cond):
    out = cond.resample(time='YS').map(
        rl.first_run,
        window=1,
        dim="time",
        coord="dayofyear",
    )
    out.attrs["units"] = ""
    return out

# TODO CE_End 最后一起复合事件的结束时间
def CE_End(cond):
    out: xarray.DataArray = cond.resample(time='YS').map(
        rl.last_run,
        window=1,
        dim="time",
        coord="dayofyear",
    )
    out.attrs["units"] = ""
    return out

# TODO CE_IS 复合事件的强度
def CE_IS2(Pr,Tem,Pr_thshold,Tem_thshold):
    Pr_Year=Pr.resample(time='YS').mean()
    Tem_Year=Tem.resample(time='YS').mean()
    IS_Pr=(Pr_Year-Pr_thshold)/Pr_thshold
    IS_Tem=(Tem_Year-Tem_thshold)/Tem_thshold
    IS_CE=(IS_Pr+IS_Tem)/2
    return IS_CE


def STD_CAL(X,additional_param):
    X_STD=np.sqrt(((X-additional_param)**2).sum(dim='time')/X.time.dt.day.shape[0])
    return X_STD
def CE_IS(Pr,Tem,Pr_thshold,Tem_thshold):
    new_func_pr = partial(STD_CAL, additional_param=Pr_thshold)
    new_func_tem = partial(STD_CAL, additional_param=Tem_thshold)
    IS_Pr=Pr.resample(time='YS').map(new_func_pr)
    IS_Tem = Tem.resample(time='YS').map(new_func_tem)
    IS_CE=(IS_Pr+IS_Tem)/2
    return IS_CE
def CE_Severity(cond):
    print(1)

def CE_Transformer(cond):
    print("有多少起是转移的复合事件")

# TODO CE_Height 每一起复合事件的高度(即每次事件发生和结束的时间之差），这里可能多张图，因为一个格点，一年可能有多起复合事件 ,即 CE_Amplitude 属于 CE_Height 的集合
def CE_Height(cond):
    print(1)
    # TODO 这两点后面加，如果有时间的话
# TODO CE_Area 每一起复合事件的面积大小,平面投影面积
def CE_Area():
    print(1)

def CE_LMF(cond,Pr_Mask,Tem_Mask):
    P_CE_actual = cond.sum(dim="time") / cond.time.dt.day.shape[0]
    P_inden=(Pr_Mask.sum(dim="time")* Tem_Mask.sum(dim="time"))/ (cond.time.dt.day.shape[0]*cond.time.dt.day.shape[0])
    LMF=P_CE_actual/P_inden
    # Return_Period[np.isinf(Return_Period)] = np.nan
    LMF.attrs["units"] = "LMF"
    return LMF



# Module=["Historical","126"]

Module=["245"]


for i in range(len(Module)):
    print(i)
    # CE_Type=["CES_CanESM5","CET_CanESM5","CETS_CanESM5"]
    CE_Type=["CETS_NESM3"]
    for j in range(len(CE_Type)):
        File_Mask = open_mfdataset(F"F:\Results\{CE_Type[j]}\\CE_Mask_{Module[i]}_*.nc", parallel=True, engine='h5netcdf', concat_dim="time",
                                   chunks={"time": 8}, combine="nested", data_vars='minimal', coords='minimal',
                                   compat='override')
        mask=File_Mask.CE_Mask

        # bbb=CE_Duration(mask)
        # CE_Duration_P = bbb.rename('CE_Duration').to_netcdf(F"E:\CE_DATA\Data_Processing\Process_Results\{CE_Type[j]}\\CE_Duration_{Module[i]}.nc", compute=False)
        # with ProgressBar():
        #     CE_Duration_P.compute()
        #
        # ccc=CE_Frequency(mask)
        # CE_Frequency_P = ccc.rename('CE_Frequency').to_netcdf(f"E:\CE_DATA\Data_Processing\Process_Results\{CE_Type[j]}\\CE_Frequency_{Module[i]}.nc", compute=False)
        # with ProgressBar():
        #     CE_Frequency_P.compute()

        ddd=CE_Amplitude(mask)
        CE_Amplitude_P = ddd.rename('CE_Amplitude').to_netcdf(F"E:\CE_DATA\Data_Processing\Process_Results\{CE_Type[j]}\\CE_Amplitude_{Module[i]}.nc", compute=False)
        with ProgressBar():
            CE_Amplitude_P.compute()

        ggg = CE_Return(mask)
        CE_Return_P = ggg.rename('CE_Return').to_netcdf(F"E:\CE_DATA\Data_Processing\Process_Results\{CE_Type[j]}\\CE_Return_{Module[i]}.nc", compute=False)
        with ProgressBar():
            CE_Return_P.compute()

        hhh = CE_Start(mask)
        CE_Start_P = hhh.rename('CE_Start').to_netcdf(F"E:\CE_DATA\Data_Processing\Process_Results\{CE_Type[j]}\\CE_Start_{Module[i]}.nc", compute=False)
        with ProgressBar():
            CE_Start_P.compute()

        iii = CE_End(mask)
        CE_End_P = iii.rename('CE_End').to_netcdf(F"E:\CE_DATA\Data_Processing\Process_Results\{CE_Type[j]}\\CE_End_{Module[i]}.nc", compute=False)
        with ProgressBar():
            CE_End_P.compute()

        File_Mask.close()

        folder_path = F"F:\Results\{CE_Type[j]}"

        search_pattern = F"CE_Mask_{Module[i]}_*.nc"

        file_paths = glob.glob(os.path.join(folder_path, search_pattern))

        for file_path in file_paths:
            os.remove(file_path)
