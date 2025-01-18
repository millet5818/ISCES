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

# Time_Base=np.arange(1950,2015,1)
FileNameList_Pr="F:\Datasets\CMIP6\CMIP6_0.15_Binear\Pr\\CMCC-ESM2\\Historical\\*.nc"
FileNameList2_Pr="F:\Datasets\CMIP6\CMIP6_0.15_Binear\Pr\\CMCC-ESM2\\126\\*.nc"
FileNameList3_Pr="F:\Datasets\CMIP6\CMIP6_0.15_Binear\Pr\\CMCC-ESM2\\245\\*.nc"
FileNameList4_Pr="F:\Datasets\CMIP6\CMIP6_0.15_Binear\Pr\\CMCC-ESM2\\585\\*.nc"

FileNameList_Tem="F:\Datasets\CMIP6\CMIP6_0.15_Binear\Max_Tem\\CMCC-ESM2\\Historical\\*.nc"
FileNameList2_Tem="F:\Datasets\CMIP6\CMIP6_0.15_Binear\Max_Tem\\CMCC-ESM2\\126\\*.nc"
FileNameList3_Tem="F:\Datasets\CMIP6\CMIP6_0.15_Binear\Max_Tem\\CMCC-ESM2\\245\\*.nc"
FileNameList4_Tem="F:\Datasets\CMIP6\CMIP6_0.15_Binear\Max_Tem\\CMCC-ESM2\\585\\*.nc"

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



# r2 = arr2.rolling(x=2, min_periods=1).max() # todo 无center是从当前点向左进行滚动，一共两个进行滚动
# r1 = arr.rolling(x=3, center=True, min_periods=1).max()# todo 表示以当前点为中心，前后采样；采取一个窗口的正确方法
# r1 = arr.rolling(x=1, center=True, min_periods=1).max()#todo 这就是时空不变采样
# Time_Base=np.arange(1950,2015,1)
# Files_Pr = open_mfdataset(FileNameList_Pr,parallel=True,concat_dim="time",chunks={'time': -1,'lat':200, 'lon': 200}, combine="nested",data_vars='minimal', coords='minimal', compat='override')
# Files_Tem = open_mfdataset(FileNameList_Tem,parallel=True,concat_dim="time",chunks={'time': -1,'lat':200, 'lon': 200}, combine="nested",data_vars='minimal', coords='minimal', compat='override')
# Files_Pr['pr'] = xclim.units.convert_units_to(Files_Pr['pr'], "mm/d")
# Files_Pr_Base=Files_Pr.where(Files_Pr.time.dt.year.isin(Time_Base),drop=True)
# PR_more = Files_Pr_Base.pr.where(Files_Pr_Base.pr >= 1)
# Pr_theshold= PR_more.chunk({"time": len(PR_more.time), "lat": 200, "lon": 200}).quantile([0.7,0.8,0.9], dim="time", keep_attrs=True)
# Files_Tem['tasmax'] = xclim.core.units.convert_units_to(Files_Tem['tasmax'], "degC")
# Files_Tem_Base=Files_Tem.where(Files_Tem.time.dt.year.isin(Time_Base),drop=True)
# Tem_theshold = Files_Tem_Base['tasmax'].chunk({"time": -1, "lat": 200, "lon": 200}).quantile([0.7,0.8,0.9], dim="time", keep_attrs=True)


Pr_theshold= xr.open_dataset("E:\CE_DATA\Data_Processing\Process_Results\Threshold\Pr\\CMCC-ESM2\\Historical_Threshold_Pr.nc").Threshold
Tem_theshold = xr.open_dataset("E:\CE_DATA\Data_Processing\Process_Results\Threshold\Tem\\CMCC-ESM2\\Historical_Threshold_Tem.nc").Threshold


Module=["Historical","126","245","585"]
fileNameList=[[FileNameList_Pr,FileNameList2_Pr,FileNameList3_Pr,FileNameList4_Pr],
              [FileNameList_Tem,FileNameList2_Tem,FileNameList3_Tem,FileNameList4_Tem]]

for i in range(len(Module)):
    # File_Pr = open_mfdataset(fileNameList[0][i],parallel=True,engine='h5netcdf',concat_dim="time",chunks={'time': -1,'lat':400, 'lon': 400}, combine="nested",data_vars='minimal', coords='minimal', compat='override')
    # File_Tem = open_mfdataset(fileNameList[1][i],parallel=True,engine='h5netcdf',concat_dim="time",chunks={'time': -1,'lat':400, 'lon': 400}, combine="nested",data_vars='minimal', coords='minimal', compat='override')
    File_Pr = open_mfdataset(fileNameList[0][i],parallel=True,engine='h5netcdf',concat_dim="time",chunks={'time': 15}, combine="nested",data_vars='minimal', coords='minimal', compat='override')
    File_Tem = open_mfdataset(fileNameList[1][i],parallel=True,engine='h5netcdf',concat_dim="time",chunks={'time':15}, combine="nested",data_vars='minimal', coords='minimal', compat='override')
    File_Pr['pr'] = xclim.core.units.convert_units_to(File_Pr['pr'], "mm/d")
    File_Tem['tasmax'] = xclim.core.units.convert_units_to(File_Tem['tasmax'], "degC")

    constrain = (">", ">=")
    cond_Pr = compare(File_Pr['pr'], ">", Pr_theshold, constrain).astype(int)
    cond_Tem = compare(File_Tem['tasmax'], ">", Tem_theshold, constrain).astype(int)

    CE_Type=["CES_CMCC-ESM2","CET_CMCC-ESM2","CETS_CMCC-ESM2"]
    for j in range(len(CE_Type)):
        print(j)
        if j ==0:
            cond_pr_roll=cond_Pr.rolling(lat=3,lon=3,center=True,min_periods=1).max()
            cond_Tem_roll=cond_Tem.rolling(lat=3,lon=3,center=True,min_periods=1).max()
        elif j ==1:
            cond_pr_roll=cond_Pr.rolling(time=3,center=True,min_periods=1).max()
            cond_Tem_roll=cond_Tem.rolling(time=3,center=True,min_periods=1).max()
        elif j==2:
            cond_pr_roll=cond_Pr.rolling(lat=3,lon=3,time=3,center=True,min_periods=1).max()
            cond_Tem_roll=cond_Tem.rolling(lat=3,lon=3,time=3,center=True,min_periods=1).max()
        # mask=np.logical_and(cond_pr_roll,cond_Tem_roll)

        mask_A=np.logical_or(np.logical_and(cond_pr_roll,cond_Tem),np.logical_and(cond_Tem_roll,cond_Pr))

        for key, group in mask_A.groupby(mask_A.time.dt.year):
                CE_Mask_P = group.rename('CE_Mask').to_netcdf(F"F:\Results\{CE_Type[j]}\\CE_Mask_{Module[i]}_{key}.nc", compute=False)
                with ProgressBar():
                    CE_Mask_P.compute()
        # todo 仅有临近的
        # Mask_Substract=np.logical_xor(mask,np.logical_and(cond_Pr,cond_Tem))

        File_Mask = open_mfdataset(F"F:\Results\{CE_Type[j]}\\CE_Mask_{Module[i]}_*.nc",
                                   parallel=True, engine='h5netcdf', concat_dim="time",
                                   chunks={"time": 8}, combine="nested", data_vars='minimal', coords='minimal',
                                   compat='override')
        mask = File_Mask.CE_Mask

        bbb=CE_Duration(mask)
        CE_Duration_P = bbb.rename('CE_Duration').to_netcdf(F"E:\CE_DATA\Data_Processing\Process_Results\{CE_Type[j]}\\CE_Duration_{Module[i]}.nc", compute=False)
        with ProgressBar():
            CE_Duration_P.compute()

        ccc=CE_Frequency(mask)
        CE_Frequency_P = ccc.rename('CE_Frequency').to_netcdf(f"E:\CE_DATA\Data_Processing\Process_Results\{CE_Type[j]}\\CE_Frequency_{Module[i]}.nc", compute=False)
        with ProgressBar():
            CE_Frequency_P.compute()

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

    File_Pr.close()
    File_Tem.close()















        #
        # eee=CE_Ratio_PR(mask,cond_pr_roll)
        # CE_Ratio_PR_P = eee.rename('CE_Ratio_PR').to_netcdf(f"E:\CE_DATA\Data_Processing\Process_Results\{CE_Type[j]}\\CE_Ratio_PR_{Module[i]}.nc", compute=False)
        # with ProgressBar():
        #     CE_Ratio_PR_P.compute()
        #
        # fff=CE_Ratio_Tem(mask,cond_Tem_roll)
        # CE_Ratio_Tem_P = fff.rename('CE_Ratio_Tem').to_netcdf(F"E:\CE_DATA\Data_Processing\Process_Results\{CE_Type[j]}\\CE_Ratio_Tem_{Module[i]}.nc", compute=False)
        # with ProgressBar():
        #     CE_Ratio_Tem_P.compute()
        #
        # jjj=CE_LMF(mask,cond_pr_roll,cond_Tem_roll)
        # CE_LMF_P = jjj.rename('CE_LMF').to_netcdf(F"E:\CE_DATA\Data_Processing\Process_Results\{CE_Type[j]}\\CE_LMF_{Module[i]}.nc", compute=False)
        # with ProgressBar():
        #     CE_LMF_P.compute()

        # kkk=CE_IS(File_Pr['pr'],File_Tem['tasmax'] ,Pr_theshold,Tem_theshold)
        # CE_IS_P = kkk.rename('CE_IS').to_netcdf(F"E:\CE_DATA\Data_Processing\Process_Results\{CE_Type[j]}\\CE_IS_{Module[i]}.nc", compute=False)
        # with ProgressBar():
        #     CE_IS_P.compute()



# arr = pram.rolling(time=window).sum(skipna=False)
# return arr.resample(time=freq).max(dim="time").assign_attrs(units=pram.units)
# Demo
# arr = xr.DataArray([[1,0,1,0,0],[1,1,1,0,0],[0,0,0,0,0],[0,0,0,1,0],[0,0,0,0,1]], dims=("x", "y"))
# r = arr.rolling(y=1, center=True, min_periods=2).max()
# r1 = arr.rolling(x=2,y=2, center=True, min_periods=1).max()
# r2 = arr.rolling(x=2,y=2, center=True, min_periods=1).max()
# mask=np.logical_and(r1,r2)
