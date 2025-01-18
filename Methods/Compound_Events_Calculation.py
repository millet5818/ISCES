"""
复合事件计算
时间上复合
空间上复合
时空复合
"""
import calendar
import pandas
import datetime
import warnings
warnings.filterwarnings("ignore")
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
import xclim.indicators
#from distributed import Client
from xclim import ensembles
import xclim.indices as xi
import xclim.core.units
from xarray import open_mfdataset
#import dask
from xclim import testing
from glob import glob
from xclim.core.calendar import percentile_doy,resample_doy
from xclim.indices.generic import threshold_count,compare,to_agg_units
from xclim.core.units import Quantified
import xclim.indices.run_length as rl
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


class CE_Index:
    def __init__(self,Files1,Files2,Var1,Var2,Method,T_Duration,S_Duration,Var1_Threshod,Var2_Threshod,Directory_CE ):
        self.Files1=Files1
        self.Files2=Files2
        self.Var1=Var1
        self.Var2=Var2
        self.Method=Method
        self.T_Duration=T_Duration
        self.S_Duration=S_Duration
        self.Var1_Threshod=Var1_Threshod
        self.Var2_Threshod=Var2_Threshod
        self.Directory_CE=Directory_CE
        self.Files1=open_mfdataset(self.Files1,concat_dim="time", combine="nested",data_vars='minimal', coords='minimal', compat='override')
        self.Files2 = open_mfdataset(self.Files2, concat_dim="time", combine="nested", data_vars='minimal',coords='minimal', compat='override')
        self.Rows=self.Files1[self.Var1].shape[1]

    # TODO 根据绝对阈值定义
    # TODO HPHTD
    def HighPrecipication_HighTemperatute_D1(self):
        print("高温强降雨复合事件重叠总天数")
        Units=['K','degC','m','mm']
        File_Type=np.array(['Temperature','Temperature','Precipitation','Precipitation'])
        warm_dnd_wet_days_ByY=[]
        warm_dnd_wet_days_ByM = []
        for i in range(self.Rows):
            print(i)
            data1 = self.Files1[self.Var1][:,i,:].load()
            data2 = self.Files2[self.Var2][:,i,:].load()
            if File_Type[np.isin(Units, self.Files1[self.Var1].units)] == "Temperature":
                data1 = xclim.core.units.convert_units_to(data1, "degC")
                data2 = xclim.core.units.amount2rate(data2, out_units="mm/d")
                self.Var2_Threshod = xclim.core.units.convert_units_to(float(self.Var2_Threshod), data2,
                                                                       context="hydro")
                self.Var1_Threshod = xclim.core.units.convert_units_to(float(self.Var1_Threshod), data1)
            else:
                data2 = xclim.core.units.convert_units_to(data2, "degC")
                data1 = xclim.core.units.amount2rate(data1, out_units="mm/d")
                self.Var1_Threshod = xclim.core.units.convert_units_to(float(self.Var1_Threshod), data1,
                                                                       context="hydro")
                self.Var2_Threshod = xclim.core.units.convert_units_to(float(self.Var2_Threshod), data2)
            pr75=data1>self.Var1_Threshod
            ht=data2>self.Var2_Threshod
            warm_and_wet_ByY = np.logical_and(ht, pr75).resample(time='YS').sum(dim="time")
            warm_and_wet_ByM = np.logical_and(ht, pr75).resample(time='MS').sum(dim="time")
            warm_and_wet_ByY.coords["time"] = pd.DatetimeIndex(warm_and_wet_ByY.coords['time']).year
            warm_and_wet_ByM.coords["time"] = pd.DatetimeIndex(warm_and_wet_ByM.coords['time']).month
            warm_and_wet_ByY.name="HPHTD1_Y"
            warm_and_wet_ByM.name = "HPHTD1_M"
            warm_and_wet_ByY.attrs["units"] = "days"
            warm_and_wet_ByM.attrs["units"] = "days"
            warm_dnd_wet_days_ByY.append(warm_and_wet_ByY)
            warm_dnd_wet_days_ByM.append(warm_and_wet_ByM)
            del warm_and_wet_ByY
            del warm_and_wet_ByM
        xr.concat(warm_dnd_wet_days_ByY, dim='latitude').to_netcdf(self.Directory_CE.split('.')[0]+"_ByY.nc")
        xr.concat(warm_dnd_wet_days_ByM, dim='latitude').to_netcdf(self.Directory_CE.split('.')[0]+"_ByM.nc")
        # xr.concat(warm_dnd_wet_days_ByY, dim='latitude').to_zarr(self.Directory_CE.split('.')[0]+"_ByY.zarr")
        # xr.concat(warm_dnd_wet_days_ByM, dim='latitude').to_zarr(self.Directory_CE.split('.')[0]+"_ByM.zarr")
        # xclim.indices.warm_and_wet_days

    # TODO 分位数以月为单位计算根据阈值定义
    def HighPrecipication_HighTemperatute_D2(self):
            print("高温强降雨复合事件重叠总天数")
            warm_dnd_wet_days_ByY=[]
            warm_dnd_wet_days_ByM = []
            for i in range(self.Rows):
                print(i)
                data1 = self.Files1[self.Var1][:,i,:].load()
                data2 = self.Files2[self.Var2][:,i,:].load()
                data1 = xclim.core.units.convert_units_to(data1, "degC")
                data2 = xclim.core.units.amount2rate(data2, out_units="mm/d")
                data2_wet = data2.where(data2 >= 1)
                das_per= percentile_doy(data1, window=5,per=float(self.Var1_Threshod)*100).sel(percentiles=float(self.Var1_Threshod)*100)
                pr_per = percentile_doy(data2_wet,window=30,per=float(self.Var2_Threshod)* 100).sel(percentiles=float(self.Var2_Threshod) * 100)
                # TODO method 1
                #warm_and_wet=xclim.indices.warm_and_wet_days(data1,data2,das_per,pr_per,freq='YS')
                # TODO method 2
                thresh1=resample_doy(das_per,data1)
                thresh2=resample_doy(pr_per,data2)
                tg75=data1>thresh1
                pr75=data2>thresh2
                warm_and_wet_ByY=np.logical_and(tg75, pr75).resample(time='YS').sum(dim="time")
                warm_and_wet_ByM = np.logical_and(tg75, pr75).resample(time='MS').sum(dim="time")
                warm_and_wet_ByY.coords["time"] = pd.DatetimeIndex(warm_and_wet_ByY.coords['time']).year
                warm_and_wet_ByM.coords["time"] = pd.DatetimeIndex(warm_and_wet_ByM.coords['time']).month
                warm_and_wet_ByY.name = "HPHTD2_Y"
                warm_and_wet_ByM.name = "HPHTD2_M"
                warm_and_wet_ByY.attrs["units"] = "days"
                warm_and_wet_ByM.attrs["units"] = "days"
                warm_dnd_wet_days_ByY.append(warm_and_wet_ByY)
                warm_dnd_wet_days_ByM.append(warm_and_wet_ByM)
                del warm_and_wet_ByY
                del warm_and_wet_ByM
            xr.concat(warm_dnd_wet_days_ByY, dim='latitude').to_netcdf(self.Directory_CE.split('.')[0] + "_ByY.nc")
            xr.concat(warm_dnd_wet_days_ByM, dim='latitude').to_netcdf(self.Directory_CE.split('.')[0] + "_ByM.nc")
            # xr.concat(warm_dnd_wet_days_ByY, dim='latitude').to_zarr(self.Directory_CE.split('.')[0] + "_ByY.zarr")
            # xr.concat(warm_dnd_wet_days_ByM, dim='latitude').to_zarr(self.Directory_CE.split('.')[0] + "_ByM.zarr")

    #TODO  分位数以年为单位计算
    def HighPrecipication_HighTemperatute_D3(self):
        warm_dnd_wet_days_ByY = []
        warm_dnd_wet_days_ByM = []
        for i in range(self.Rows):
            print(i)
            data1 = self.Files1[self.Var1][:, i, :].load()
            data2 = self.Files2[self.Var2][:, i, :].load()
            data1 = xclim.core.units.convert_units_to(data1, "degC")
            data2 = xclim.core.units.amount2rate(data2, out_units="mm/d")
            data2_wet = data2.where(data2 >= 1)
            das_per1=data1.quantile([float(self.Var1_Threshod)], dim='time',keep_attrs=True).drop('quantile')
            threshod = data2_wet.quantile([float(self.Var2_Threshod)], dim='time',keep_attrs=True).drop('quantile')
            tg75=data1>das_per1
            pr75=data2>threshod
            warm_and_wet_ByY=np.logical_and(tg75, pr75).resample(time='YS').sum(dim="time")
            warm_and_wet_ByM = np.logical_and(tg75, pr75).resample(time='MS').sum(dim="time")
            warm_and_wet_ByY.coords["time"] = pd.DatetimeIndex(warm_and_wet_ByY.coords['time']).year
            warm_and_wet_ByM.coords["time"] = pd.DatetimeIndex(warm_and_wet_ByM.coords['time']).month
            warm_and_wet_ByY.name = "HPHTD3_Y"
            warm_and_wet_ByM.name = "HPHTD3_M"
            warm_and_wet_ByY.attrs["units"] = "days"
            warm_and_wet_ByM.attrs["units"] = "days"
            warm_dnd_wet_days_ByY.append(warm_and_wet_ByY)
            warm_dnd_wet_days_ByM.append(warm_and_wet_ByM)
            del warm_and_wet_ByY
            del warm_and_wet_ByM
        xr.concat(warm_dnd_wet_days_ByY, dim='latitude').to_netcdf(self.Directory_CE.split('.')[0] + "_ByY.nc")
        xr.concat(warm_dnd_wet_days_ByM, dim='latitude').to_netcdf(self.Directory_CE.split('.')[0] + "_ByM.nc")
        # xr.concat(warm_dnd_wet_days_ByY, dim='latitude').to_zarr(self.Directory_CE.split('.')[0] + "_ByY.zarr")
        # xr.concat(warm_dnd_wet_days_ByM, dim='latitude').to_zarr(self.Directory_CE.split('.')[0] + "_ByM.zarr")

        # xclim.indices.warm_and_wet_days
    # TODO 属于绝对阈值的方法计算
    def HighPrecipication_HighTemperatute_D4(self):
        "至少连续几（这里设置的是3）天，温度和降雨都大于某一阈值的最长序列, 被视为一次高温和强降雨事件,求筛选下来所有事件的总长度 window设置2,based on hot_spell_max_length 源码"
        HPHTD4 = []
        for i in range(self.Rows):
            print(i)
            data1 = self.Files1[self.Var1][:, i, :].load()
            data2 = self.Files2[self.Var2][:, i, :].load()
            data1 = xclim.core.units.convert_units_to(data1, "degC")
            data2 = xclim.core.units.amount2rate(data2, out_units="mm/d")
            cond1 = data1 > xclim.core.units.convert_units_to(float(self.Var1_Threshod), data1)
            cond2 = data2 > xclim.core.units.convert_units_to(float(self.Var2_Threshod), data2, context="hydro")
            cond = np.logical_and(cond1, cond2)
            max_l = rl.resample_and_rl(cond, True, rl.windowed_run_count, window=1, freq='YS')
            out = max_l.where(max_l >= 3, 0)
            out.coords["time"] = pd.DatetimeIndex(out.coords['time']).year
            out.name = "HPHTD4"
            HPHTD4.append(out)
            del out
        xr.concat(HPHTD4, dim='latitude').to_netcdf(self.Directory_CE)


    #TODO HPHTF1 绝对阈值 并且逐年保存数据--适用于内存小的电脑
    def HighPrecipication_HighTemperatute_F1(self):
        print("高温强降雨复合事件重叠分布")
        Units = ['K', 'degC', 'm', 'mm']
        File_Type = np.array(['Temperature', 'Temperature', 'Precipitation', 'Precipitation'])
        for t in np.unique(pd.DatetimeIndex(self.Files1.coords['time']).year):
            print(t)
            data1 = self.Files1[self.Var1].sel(time=(self.Files1.time.dt.year==t)).load()
            data2 = self.Files2[self.Var2].sel(time=(self.Files2.time.dt.year==t)).load()
            if File_Type[np.isin(Units, self.Files1[self.Var1].units)] == "Temperature":
                data1 = xclim.core.units.convert_units_to(data1, "degC")
                data2 = xclim.core.units.amount2rate(data2, out_units="mm/d")
                self.Var2_Threshod = xclim.core.units.convert_units_to(float(self.Var2_Threshod), data2,
                                                                       context="hydro")
                self.Var1_Threshod = xclim.core.units.convert_units_to(float(self.Var1_Threshod), data1)
            else:
                data2 = xclim.core.units.convert_units_to(data2, "degC")
                data1 = xclim.core.units.amount2rate(data1, out_units="mm/d")
                self.Var1_Threshod = xclim.core.units.convert_units_to(float(self.Var1_Threshod), data1,
                                                                       context="hydro")
                self.Var2_Threshod = xclim.core.units.convert_units_to(float(self.Var2_Threshod), data2)
            pr75 = data1 > self.Var1_Threshod
            ht = data2 > self.Var2_Threshod
            # list(warm_and_wet.groupby(warm_and_wet.time.dt.year))
            warm_and_wet = np.logical_and(ht, pr75).astype(int)
            warm_and_wet.name = "HPHTF1"
            warm_and_wet.to_netcdf(self.Directory_CE.split('.')[0]+"_"+str(t)+'.nc')

            del warm_and_wet, data1, data2

    # TODO HPHTF2 绝对阈值 并且一次性保存所有年份数据--适用于内存大的电脑
    def HighPrecipication_HighTemperatute_F2(self):
        print("高温强降雨复合事件重叠分布")
        Units = ['K', 'degC', 'm', 'mm']
        File_Type = np.array(['Temperature', 'Temperature', 'Precipitation', 'Precipitation'])
        warm_dnd_wet_days = []
        for i in range(self.Rows):
            print(i)
            data1 = self.Files1[self.Var1][:, i, :].load()
            data2 = self.Files2[self.Var2][:, i, :].load()
            if File_Type[np.isin(Units, self.Files1[self.Var1].units)] == "Temperature":
                data1 = xclim.core.units.convert_units_to(data1, "degC")
                data2 = xclim.core.units.amount2rate(data2, out_units="mm/d")
                self.Var2_Threshod = xclim.core.units.convert_units_to(float(self.Var2_Threshod), data2,
                                                                       context="hydro")
                self.Var1_Threshod = xclim.core.units.convert_units_to(float(self.Var1_Threshod), data1)
            else:
                data2 = xclim.core.units.convert_units_to(data2, "degC")
                data1 = xclim.core.units.amount2rate(data1, out_units="mm/d")
                self.Var1_Threshod = xclim.core.units.convert_units_to(float(self.Var1_Threshod), data1,
                                                                       context="hydro")
                self.Var2_Threshod = xclim.core.units.convert_units_to(float(self.Var2_Threshod), data2)
            pr75 = data1 > self.Var1_Threshod
            ht = data2 > self.Var2_Threshod
            # list(warm_and_wet.groupby(warm_and_wet.time.dt.year))
            warm_and_wet = np.logical_and(ht, pr75).astype(int)
            warm_and_wet.name = "HPHTF2"
            warm_dnd_wet_days.append(warm_and_wet)
            del warm_and_wet
        xr.concat(warm_dnd_wet_days, dim='latitude').to_netcdf(self.Directory_CE)

    #TODO HPHTF2 相对阈值1 并且逐年保存数据--适用于内存小的电脑
    def HighPrecipication_HighTemperatute_F3(self):
        print("高温强降雨复合事件重叠分布")
        for t in np.unique(pd.DatetimeIndex(self.Files1.coords['time']).year):
            print(t)
            data1 = self.Files1[self.Var1].sel(time=(self.Files1.time.dt.year==t)).load()
            data2 = self.Files2[self.Var2].sel(time=(self.Files2.time.dt.year==t)).load()
            data2_wet = data2.where(data2 >= 1)
            das_per = percentile_doy(data1, window=5, per=float(self.Var1_Threshod) * 100).sel(
                percentiles=float(self.Var1_Threshod) * 100)
            pr_per = percentile_doy(data2_wet, window=30, per=float(self.Var2_Threshod) * 100).sel(
                percentiles=float(self.Var2_Threshod) * 100)
            # TODO method 1
            # warm_and_wet=xclim.indices.warm_and_wet_days(data1,data2,das_per,pr_per,freq='YS')
            # TODO method 2
            thresh1 = resample_doy(das_per, data1)
            thresh2 = resample_doy(pr_per, data2)
            tg75 = data1 > thresh1
            pr75 = data2 > thresh2
            warm_and_wet = np.logical_and(tg75, pr75).astype(int)
            warm_and_wet.name = "HPHTF3"
            warm_and_wet.to_netcdf(self.Directory_CE.split('.')[0]+"_"+str(t)+'.nc')
            del warm_and_wet, data1, data2

    # TODO HPHTF4 相对阈值1 并且一次性保存所有年份数据--适用于内存大的电脑
    def HighPrecipication_HighTemperatute_F4(self):
        print("高温强降雨复合事件重叠分布")
        Units = ['K', 'degC', 'm', 'mm']
        File_Type = np.array(['Temperature', 'Temperature', 'Precipitation', 'Precipitation'])
        warm_dnd_wet_days = []
        for i in range(self.Rows):
            print(i)
            data1 = self.Files1[self.Var1][:, i, :].load()
            data2 = self.Files2[self.Var2][:, i, :].load()
            data2_wet = data2.where(data2 >= 1)
            das_per = percentile_doy(data1, window=5, per=float(self.Var1_Threshod) * 100).sel(
                percentiles=float(self.Var1_Threshod) * 100)
            pr_per = percentile_doy(data2_wet, window=30, per=float(self.Var2_Threshod) * 100).sel(
                percentiles=float(self.Var2_Threshod) * 100)
            # TODO method 1
            # warm_and_wet=xclim.indices.warm_and_wet_days(data1,data2,das_per,pr_per,freq='YS')
            # TODO method 2
            thresh1 = resample_doy(das_per, data1)
            thresh2 = resample_doy(pr_per, data2)
            tg75 = data1 > thresh1
            pr75 = data2 > thresh2
            warm_and_wet = np.logical_and(tg75, pr75).astype(int)
            warm_and_wet.name = "HPHT4"
            warm_dnd_wet_days.append(warm_and_wet)
            del warm_and_wet
        xr.concat(warm_dnd_wet_days, dim='latitude').to_netcdf(self.Directory_CE)

    #TODO HPHTF3 相对阈值1 并且逐年保存数据--适用于内存小的电脑
    def HighPrecipication_HighTemperatute_F5(self):
        print("高温强降雨复合事件重叠分布")
        for t in np.unique(pd.DatetimeIndex(self.Files1.coords['time']).year):
            print(t)
            data1 = self.Files1[self.Var1].sel(time=(self.Files1.time.dt.year==t)).load()
            data2 = self.Files2[self.Var2].sel(time=(self.Files2.time.dt.year==t)).load()
            data1 = xclim.core.units.convert_units_to(data1, "degC")
            data2 = xclim.core.units.amount2rate(data2, out_units="mm/d")
            data2_wet = data2.where(data2 >= 1)
            thresh1 = data1.quantile([float(self.Var1_Threshod)], dim='time', keep_attrs=True)
            thresh2 = data2_wet.quantile([float(self.Var2_Threshod)], dim='time', keep_attrs=True)
            tg75 = data1 > thresh1
            pr75 = data2 > thresh2
            warm_and_wet = np.logical_and(tg75, pr75).astype(int)
            warm_and_wet.name = "HPHTF5"
            warm_and_wet.to_netcdf(self.Directory_CE.split('.')[0]+"_"+str(t)+'.nc')
            del warm_and_wet, data1, data2

    # TODO HPHTF6 相对阈值2 并且一次性保存所有年份数据--适用于内存大的电脑
    def HighPrecipication_HighTemperatute_F6(self):
        print("高温强降雨复合事件重叠分布")
        warm_dnd_wet_days = []
        for i in range(self.Rows):
            print(i)
            data1 = self.Files1[self.Var1][:, i, :].load()
            data2 = self.Files2[self.Var2][:, i, :].load()
            data2_wet = data2.where(data2 >= 1)
            thresh1 = data1.quantile([float(self.Var1_Threshod)], dim='time', keep_attrs=True)
            thresh2 = data2_wet.quantile([float(self.Var2_Threshod)], dim='time', keep_attrs=True)
            tg75 = data1 > thresh1
            pr75 = data2 > thresh2
            warm_and_wet = np.logical_and(tg75, pr75).astype(int)
            warm_and_wet.name = "HPHT6"
            warm_dnd_wet_days.append(warm_and_wet)
            del warm_and_wet
        xr.concat(warm_dnd_wet_days, dim='latitude').to_netcdf(self.Directory_CE)


    # TODO HPHTM1,绝对阈值，复合事件的最大重叠长度
    def HighPrecipication_HighTemperatute_MaxLength1(self):
        print("高温强降雨复合事件重叠分布单次重叠最大持续长度DAY")
        "至少连续几（这里设置的是2）天，温度和降雨都大于某一阈值的最长序列,window设置2,based on hot_spell_max_length 源码"
        HPHTM_Y=[]
        HPHTM_M = []
        for i in range(self.Rows):
            print(i)
            data1 = self.Files1[self.Var1][:, i, :].load()
            data2 = self.Files2[self.Var2][:, i, :].load()
            data1 = xclim.core.units.convert_units_to(data1, "degC")
            data2 = xclim.core.units.amount2rate(data2, out_units="mm/d")
            cond1 = data1>xclim.core.units.convert_units_to(float(self.Var1_Threshod), data1)
            cond2 = data2>xclim.core.units.convert_units_to(float(self.Var2_Threshod), data2,context="hydro")
            cond=np.logical_and(cond1,cond2)

            max_l = rl.resample_and_rl(cond,True,rl.longest_run,freq='YS')
            out = max_l.where(max_l >= 2, 0)
            out.coords["time"] = pd.DatetimeIndex(out.coords['time']).year
            out.name = "HPHTM_Y"
            out.attrs["units"] = "days"

            max_l_M = rl.resample_and_rl(cond,True,rl.longest_run,freq='MS')
            out_M = max_l_M.where(max_l_M >= 2, 0)
            out_M.coords["time"] = pd.DatetimeIndex(out_M.coords['time']).month
            out_M.name = "HPHTM_M"
            out_M.attrs["units"] = "days"

            HPHTM_Y.append(out)
            HPHTM_M.append(out_M)
            del out
            del out_M
        xr.concat(HPHTM_Y, dim='latitude').to_netcdf(self.Directory_CE.split('.')[0]+"_ByY.nc")
        xr.concat(HPHTM_M, dim='latitude').to_netcdf(self.Directory_CE.split('.')[0] + "_ByM.nc")
        # xr.concat(HPHTM_Y, dim='latitude').to_zarr(self.Directory_CE.split('.')[0]+"_ByY.zarr")
        # xr.concat(HPHTM_M, dim='latitude').to_zarr(self.Directory_CE.split('.')[0] + "_ByM.zarr")


    # TODO HPHTM2,相对阈值1，复合事件的最大重叠长度
    def HighPrecipication_HighTemperatute_MaxLength2(self):
        print("高温强降雨复合事件重叠分布单次重叠最大持续长度DAY")
        "至少连续几（这里设置的是2）天，温度和降雨都大于某一阈值的最长序列,window设置2,based on hot_spell_max_length 源码"
        HPHTM_Y=[]
        HPHTM_M = []
        for i in range(self.Rows):
            print(i)
            data1 = self.Files1[self.Var1][:, i, :].load()
            data2 = self.Files2[self.Var2][:, i, :].load()
            data1 = xclim.core.units.convert_units_to(data1, "degC")
            data2 = xclim.core.units.amount2rate(data2, out_units="mm/d")
            data2_wet = data2.where(data2 >= 1)
            das_per = percentile_doy(data1, window=5, per=float(self.Var1_Threshod) * 100).sel(
                percentiles=float(self.Var1_Threshod) * 100)
            pr_per = percentile_doy(data2_wet, window=30, per=float(self.Var2_Threshod) * 100).sel(
                percentiles=float(self.Var2_Threshod) * 100)
            thresh1 = resample_doy(das_per, data1)
            thresh2 = resample_doy(pr_per, data2)
            cond1 = data1 > thresh1
            cond2 = data2 > thresh2
            cond  = np.logical_and(cond1,cond2)
            max_l = rl.resample_and_rl(cond,True,rl.longest_run,freq='YS')
            out = max_l.where(max_l >= 2, 0)
            out.coords["time"] = pd.DatetimeIndex(out.coords['time']).year
            out.name = "HPHTM2_Y"
            out.attrs["units"] = "days"
            HPHTM_Y.append(out)

            max_l_M = rl.resample_and_rl(cond, True, rl.longest_run, freq='MS')
            out_M = max_l_M.where(max_l_M >= 2, 0)
            out_M.coords["time"] = pd.DatetimeIndex(out_M.coords['time']).month
            out_M.name = "HPHTM2_M"
            out_M.attrs["units"] = "days"
            HPHTM_M.append(out_M)
            del max_l
            del max_l_M
        xr.concat(HPHTM_Y, dim='latitude').to_netcdf(self.Directory_CE.split('.')[0]+"_ByY.nc")
        xr.concat(HPHTM_M, dim='latitude').to_netcdf(self.Directory_CE.split('.')[0]+"_ByM.nc")
        # xr.concat(HPHTM_Y, dim='latitude').to_zarr(self.Directory_CE.split('.')[0]+"_ByY.zarr")
        # xr.concat(HPHTM_M, dim='latitude').to_zarr(self.Directory_CE.split('.')[0]+"_ByM.zarr")


    # TODO HPHTM3,相对阈值2，复合事件的最大重叠长度
    def HighPrecipication_HighTemperatute_MaxLength3(self):
        print("高温强降雨复合事件重叠分布单次重叠最大持续长度DAY")
        "至少连续几（这里设置的是2）天，温度和降雨都大于某一阈值的最长序列,window设置2,based on hot_spell_max_length 源码"
        HPHTM_Y = []
        HPHTM_M = []
        for i in range(self.Rows):
            print(i)
            data1 = self.Files1[self.Var1][:, i, :].load()
            data2 = self.Files2[self.Var2][:, i, :].load()
            data1 = xclim.core.units.convert_units_to(data1, "degC")
            data2 = xclim.core.units.amount2rate(data2, out_units="mm/d")
            data2_wet = data2.where(data2 >= 1)
            # thresh1 = data1.quantile([float(self.Var1_Threshod)], dim='time', keep_attrs=True)
            # thresh2 = data2_wet.quantile([float(self.Var2_Threshod)], dim='time', keep_attrs=True)
            thresh1 = data1.quantile([float(self.Var1_Threshod)], dim='time', keep_attrs=True).drop('quantile')
            thresh2 = data2_wet.quantile([float(self.Var2_Threshod)], dim='time', keep_attrs=True).drop('quantile')
            # da.sel(x=0, y=0) # 基于坐标轴的标签或值
            # da.isel(x=0, y=0)  # 基于坐标轴索引
            # ds.drop_dims("time")
            # ds.drop_sel(quantile=[1])
            cond1 = data1 > thresh1
            cond2 = data2 > thresh2
            cond = np.logical_and(cond1, cond2)
            max_l = rl.resample_and_rl(cond, True, rl.longest_run, freq='YS')
            out = max_l.where(max_l >= 2, 0)
            out.coords["time"] = pd.DatetimeIndex(out.coords['time']).year
            out.name = "HPHTM3_Y"
            out.attrs["units"] = "days"
            HPHTM_Y.append(out)
            max_l_M = rl.resample_and_rl(cond, True, rl.longest_run, freq='MS')
            out_M = max_l_M.where(max_l_M >= 2, 0)
            out_M.coords["time"] = pd.DatetimeIndex(out_M.coords['time']).month
            out_M.name = "HPHTM3_M"
            out_M.attrs["units"] = "days"
            HPHTM_M.append(out_M)
            del out
        xr.concat(HPHTM_Y, dim='latitude').to_netcdf(self.Directory_CE.split('.')[0] + "_ByY.nc")
        xr.concat(HPHTM_M, dim='latitude').to_netcdf(self.Directory_CE.split('.')[0] + "_ByM.nc")
    #     # xr.concat(HPHTM_Y, dim='latitude').to_dataset().to_zarr(self.Directory_CE.split('.')[0] + "_ByY.zarr")
    #     # xr.concat(HPHTM_M, dim='latitude').to_dataset().to_zarr(self.Directory_CE.split('.')[0] + "_ByM.zarr")

    # TODO HPHTEN1 绝对阈值，复合事件发生的次数
    def HighPrecipication_HighTemperatute_EN1(self):
        print("高温强降雨复合事件重叠分布事件次数")
        "至少连续几（这里设置的是3）天，温度和降雨都大于某一阈值的最长序列, 被视为一次高温和强降雨事件,求筛选下来所有事件的总长度 window设置2,based on hot_spell_max_length 源码"
        HPHTEN_Y = []
        HPHTEN_M = []
        for i in range(self.Rows):
            print(i)
            data1 = self.Files1[self.Var1][:, i, :].load()
            data2 = self.Files2[self.Var2][:, i, :].load()
            data1 = xclim.core.units.convert_units_to(data1, "degC")
            data2 = xclim.core.units.amount2rate(data2, out_units="mm/d")
            cond1 = data1 > xclim.core.units.convert_units_to(float(self.Var1_Threshod), data1)
            cond2 = data2 > xclim.core.units.convert_units_to(float(self.Var2_Threshod), data2, context="hydro")
            cond = np.logical_and(cond1, cond2)


            out = rl.resample_and_rl(cond, True, rl.windowed_run_events,window=1, freq='YS')
            out.coords["time"] = pd.DatetimeIndex(out.coords['time']).year
            out.name = "HPHTEN1_Y"
            out.attrs["units"] = ""
            HPHTEN_Y.append(out)

            out_M = rl.resample_and_rl(cond, True, rl.windowed_run_events,window=1, freq='MS')
            out_M.coords["time"] = pd.DatetimeIndex(out_M.coords['time']).month
            out_M.name = "HPHTEN1_M"
            out_M.attrs["units"] = ""
            HPHTEN_M.append(out_M)
            del out
            del out_M

        xr.concat(HPHTEN_Y, dim='latitude').to_netcdf(self.Directory_CE.split('.')[0] + "_ByY.nc")
        xr.concat(HPHTEN_M, dim='latitude').to_netcdf(self.Directory_CE.split('.')[0] + "_ByM.nc")
        # xr.concat(HPHTEN_Y, dim='latitude').to_zarr(self.Directory_CE.split('.')[0] + "_ByY.zarr")
        # xr.concat(HPHTEN_M, dim='latitude').to_zarr(self.Directory_CE.split('.')[0] + "_ByM.zarr")


    # TODO HPHTEN2 相对阈值1，复合事件发生的次数
    def HighPrecipication_HighTemperatute_EN2(self):
        print("高温强降雨复合事件重叠分布事件次数")
        "至少连续几（这里设置的是3）天，温度和降雨都大于某一阈值的最长序列, 被视为一次高温和强降雨事件,求筛选下来所有事件的总长度 window设置2,based on hot_spell_max_length 源码"
        HPHTEN_Y = []
        HPHTEN_M = []
        for i in range(self.Rows):
            print(i)
            data1 = self.Files1[self.Var1][:, i, :].load()
            data2 = self.Files2[self.Var2][:, i, :].load()
            data1 = xclim.core.units.convert_units_to(data1, "degC")
            data2 = xclim.core.units.amount2rate(data2, out_units="mm/d")
            data2_wet = data2.where(data2 >= 1)
            das_per = percentile_doy(data1, window=5, per=float(self.Var1_Threshod) * 100).sel(
                percentiles=float(self.Var1_Threshod) * 100)
            pr_per = percentile_doy(data2_wet, window=30, per=float(self.Var2_Threshod) * 100).sel(
                percentiles=float(self.Var2_Threshod) * 100)
            thresh1 = resample_doy(das_per, data1)
            thresh2 = resample_doy(pr_per, data2)
            cond1 = data1 > thresh1
            cond2 = data2 > thresh2
            cond = np.logical_and(cond1, cond2)
            out = rl.resample_and_rl(cond, True, rl.windowed_run_events,window=1, freq='YS')
            out.coords["time"] = pd.DatetimeIndex(out.coords['time']).year
            out.name = "HPHTEN2_Y"
            out.attrs["units"] = ""
            HPHTEN_Y.append(out)

            out_M = rl.resample_and_rl(cond, True, rl.windowed_run_events, window=1, freq='MS')
            out_M.coords["time"] = pd.DatetimeIndex(out_M.coords['time']).month
            out_M.name = "HPHTEN2_M"
            out_M.attrs["units"] = ""
            HPHTEN_M.append(out_M)

            del out
        xr.concat(HPHTEN_Y, dim='latitude').to_netcdf(self.Directory_CE.split('.')[0] + "_ByY.nc")
        xr.concat(HPHTEN_M, dim='latitude').to_netcdf(self.Directory_CE.split('.')[0] + "_ByM.nc")
        # xr.concat(HPHTEN_Y, dim='latitude').to_zarr(self.Directory_CE.split('.')[0] + "_ByY.zarr")
        # xr.concat(HPHTEN_M, dim='latitude').to_zarr(self.Directory_CE.split('.')[0] + "_ByM.zarr")


    # TODO HPHTEN3 相对阈值2，复合事件发生的次数
    def HighPrecipication_HighTemperatute_EN3(self):
        print("高温强降雨复合事件重叠分布事件次数")
        "至少连续几（这里设置的是3）天，温度和降雨都大于某一阈值的最长序列, 被视为一次高温和强降雨事件,求筛选下来所有事件的总长度 window设置2,based on hot_spell_max_length 源码"
        HPHTEN_Y = []
        HPHTEN_M = []
        for i in range(self.Rows):
            print(i)
            data1 = self.Files1[self.Var1][:, i, :].load()
            data2 = self.Files2[self.Var2][:, i, :].load()
            data1 = xclim.core.units.convert_units_to(data1, "degC")
            data2 = xclim.core.units.amount2rate(data2, out_units="mm/d")
            data2_wet = data2.where(data2 >= 1)
            thresh1 = data1.quantile([float(self.Var1_Threshod)], dim='time', keep_attrs=True).drop('quantile')
            thresh2 = data2_wet.quantile([float(self.Var2_Threshod)], dim='time', keep_attrs=True).drop('quantile')
            cond1 = data1 > thresh1
            cond2 = data2 > thresh2
            cond = np.logical_and(cond1, cond2)

            out = rl.resample_and_rl(cond, True, rl.windowed_run_events,window=1, freq='YS')
            out.coords["time"] = pd.DatetimeIndex(out.coords['time']).year
            out.name = "HPHTEN3_Y"
            out.attrs["units"] = ""
            HPHTEN_Y.append(out)

            out_M = rl.resample_and_rl(cond, True, rl.windowed_run_events, window=1, freq='MS')
            out_M.coords["time"] = pd.DatetimeIndex(out_M.coords['time']).month
            out_M.name = "HPHTEN3_M"
            out_M.attrs["units"] = ""
            HPHTEN_M.append(out_M)
            del out
        xr.concat(HPHTEN_Y, dim='latitude').to_netcdf(self.Directory_CE.split('.')[0] + "_ByY.nc")
        xr.concat(HPHTEN_M, dim='latitude').to_netcdf(self.Directory_CE.split('.')[0] + "_ByM.nc")
        # xr.concat(HPHTEN_Y, dim='latitude').to_zarr(self.Directory_CE.split('.')[0] + "_ByY.zarr")
        # xr.concat(HPHTEN_M, dim='latitude').to_zarr(self.Directory_CE.split('.')[0] + "_ByM.zarr")

    # TODO HPHTRA1,采用绝对阈值 复合事件占据极端高温事件的年百分比，也可叫做severty严重程度，即每个网格一年复合事件发生的天数/每个网格一年发生极端高温事件的天数；x=100(N11/(N11+N21))
    def HighPrecipication_HighTemperatute_RA1(self):
        print("HPHTRA1")
        print("高温强降雨复合事件重叠总天数")
        Units = ['K', 'degC', 'm', 'mm']
        File_Type = np.array(['Temperature', 'Temperature', 'Precipitation', 'Precipitation'])
        HPHTRA1 = []
        Return_Period=[]
        for i in range(self.Rows):
            print(i)
            data1 = self.Files1[self.Var1][:, i, :].load()
            data2 = self.Files2[self.Var2][:, i, :].load()
            if File_Type[np.isin(Units, self.Files1[self.Var1].units)] == "Temperature":
                data1 = xclim.core.units.convert_units_to(data1, "degC")
                data2 = xclim.core.units.amount2rate(data2, out_units="mm/d")
                self.Var2_Threshod = xclim.core.units.convert_units_to(float(self.Var2_Threshod), data2,
                                                                       context="hydro")
                self.Var1_Threshod = xclim.core.units.convert_units_to(float(self.Var1_Threshod), data1)
            else:
                data2 = xclim.core.units.convert_units_to(data2, "degC")
                data1 = xclim.core.units.amount2rate(data1, out_units="mm/d")
                self.Var1_Threshod = xclim.core.units.convert_units_to(float(self.Var1_Threshod), data1,
                                                                       context="hydro")
                self.Var2_Threshod = xclim.core.units.convert_units_to(float(self.Var2_Threshod), data2)
            pr75 = data1 > self.Var1_Threshod
            ht = data2 > self.Var2_Threshod
            warm_and_wet = np.logical_and(ht, pr75)
            warm_and_wet_year =warm_and_wet.resample(time='YS').sum(dim="time")
            warm_and_wet_allyear = warm_and_wet.sum(dim="time")
            warm = pr75.resample(time='YS').sum(dim="time")
            wet=ht.resample(time='YS').sum(dim="time")
            HPHTRA_P = warm_and_wet_year / warm
            HPHTRA_T = warm_and_wet_year / wet
            # TODO Joint occurrence probability for each grid cell by adding the number of all joint exceedances in this grid cell and dividing it by the total number of days in the study period.
            # TODO Probability map of joint exceedance = sum of binary map over time /the number of days in the study periods
            PXY=warm_and_wet_allyear/warm_and_wet.time.dt.day.shape[0]

            # todo The Return period[RP with [RP] =years] is then Calculated as 1/365*PXY
            RP_XY=1/(PXY*365)
            RP_XY[np.isinf(RP_XY)]=np.nan
            HPHTRA_P.coords["time"] = pd.DatetimeIndex(HPHTRA_P.coords['time']).year
            HPHTRA_T.coords["time"] = pd.DatetimeIndex(HPHTRA_T.coords['time']).year
            HPHTRA_P.attrs["units"] = "%"
            HPHTRA_T.attrs["units"] = "%"
            a=xr.merge([HPHTRA_P.rename("HPHTRA_P"),HPHTRA_T.rename("HPHTRA_T")])
            HPHTRA1.append(a)
            RP_XY.name = "Return_Period"
            RP_XY.attrs["units"] = "yrs"
            Return_Period.append(RP_XY)
            del warm_and_wet,warm,wet,a
        xr.concat(HPHTRA1, dim='latitude').to_netcdf(self.Directory_CE)
        xr.concat(Return_Period, dim='latitude').to_netcdf(self.Directory_CE.split('.')[0]+"_Return_Period.nc")
        # xr.concat(HPHTRA1, dim='latitude').to_zarr(self.Directory_CE)
        # xr.concat(Return_Period, dim='latitude').tozarr(self.Directory_CE.split('.')[0]+"_Return_Period.zarr")

    # TODO HPHTRA2,采用相对阈值1 复合事件占据极端高温事件的年百分比，也可叫做severty严重程度，即每个网格一年复合事件发生的天数/每个网格一年发生极端高温事件的天数；x=100(N11/(N11+N21))
    def HighPrecipication_HighTemperatute_RA2(self):
        print("HPHTRA2")
        HPHTRA1 = []
        Return_Period=[]
        for i in range(self.Rows):
            print(i)
            data1 = self.Files1[self.Var1][:, i, :].load()
            data2 = self.Files2[self.Var2][:, i, :].load()
            data1 = xclim.core.units.convert_units_to(data1, "degC")
            data2 = xclim.core.units.amount2rate(data2, out_units="mm/d")
            data2_wet = data2.where(data2 >= 1)
            das_per = percentile_doy(data1, window=5, per=float(self.Var1_Threshod) * 100).sel(
                percentiles=float(self.Var1_Threshod) * 100)
            pr_per = percentile_doy(data2_wet, window=30, per=float(self.Var2_Threshod) * 100).sel(
                percentiles=float(self.Var2_Threshod) * 100)
            thresh1 = resample_doy(das_per, data1)
            thresh2 = resample_doy(pr_per, data2)
            cond1 = data1 > thresh1
            cond2 = data2 > thresh2
            warm_and_wet = np.logical_and(cond1, cond2)
            warm_and_wet_year =warm_and_wet.resample(time='YS').sum(dim="time")
            warm_and_wet_allyear = warm_and_wet.sum(dim="time")
            warm = cond1.resample(time='YS').sum(dim="time")
            wet=cond2.resample(time='YS').sum(dim="time")
            HPHTRA_P = warm_and_wet_year / warm
            HPHTRA_T = warm_and_wet_year / wet
            # TODO Joint occurrence probability for each grid cell by adding the number of all joint exceedances in this grid cell and dividing it by the total number of days in the study period.
            # TODO Probability map of joint exceedance = sum of binary map over time /the number of days in the study periods
            PXY=warm_and_wet_allyear/warm_and_wet.time.dt.day.shape[0]
            # todo The Return period[RP with [RP] =years] is then Calculated as 1/365*PXY
            RP_XY=1/(PXY*365)
            HPHTRA_P.coords["time"] = pd.DatetimeIndex(HPHTRA_P.coords['time']).year
            HPHTRA_T.coords["time"] = pd.DatetimeIndex(HPHTRA_T.coords['time']).year
            HPHTRA_P.attrs["units"] = "%"
            HPHTRA_T.attrs["units"] = "%"
            a=xr.merge([HPHTRA_P.rename("HPHTRA_P"),HPHTRA_T.rename("HPHTRA_T")])
            HPHTRA1.append(a)
            RP_XY.name = "Return_Period"
            RP_XY.attrs["units"] = "yrs"
            Return_Period.append(RP_XY)
            del warm_and_wet,warm,wet,a
        xr.concat(HPHTRA1, dim='latitude').to_netcdf(self.Directory_CE)
        xr.concat(Return_Period, dim='latitude').to_netcdf(self.Directory_CE.split('.')[0]+"_Return_Period.nc")
        # xr.concat(HPHTRA1, dim='latitude').to_zarr(self.Directory_CE)
        # xr.concat(Return_Period, dim='latitude').to_zarr(self.Directory_CE.split('.')[0]+"_Return_Period.zarr")

    def HighPrecipication_HighTemperatute_RA3(self):
        print("HPHTRA3")
        HPHTRA1 = []
        Return_Period = []
        for i in range(self.Rows):
            print(i)
            data1 = self.Files1[self.Var1][:, i, :].load()
            data2 = self.Files2[self.Var2][:, i, :].load()
            data1 = xclim.core.units.convert_units_to(data1, "degC")
            data2 = xclim.core.units.amount2rate(data2, out_units="mm/d")
            data2_wet = data2.where(data2 >= 1)
            thresh1 = data1.quantile([float(self.Var1_Threshod)], dim='time', keep_attrs=True).drop('quantile')
            thresh2 = data2_wet.quantile([float(self.Var2_Threshod)], dim='time', keep_attrs=True).drop('quantile')
            cond1 = data1 > thresh1
            cond2 = data2 > thresh2
            warm_and_wet = np.logical_and(cond1, cond2)
            warm_and_wet_year = warm_and_wet.resample(time='YS').sum(dim="time")
            warm_and_wet_allyear = warm_and_wet.sum(dim="time")
            warm = cond1.resample(time='YS').sum(dim="time")
            wet = cond2.resample(time='YS').sum(dim="time")
            HPHTRA_P = warm_and_wet_year / warm
            HPHTRA_T = warm_and_wet_year / wet
            # TODO Joint occurrence probability for each grid cell by adding the number of all joint exceedances in this grid cell and dividing it by the total number of days in the study period.
            # TODO Probability map of joint exceedance = sum of binary map over time /the number of days in the study periods
            PXY = warm_and_wet_allyear / warm_and_wet.time.dt.day.shape[0]
            # todo The Return period[RP with [RP] =years] is then Calculated as 1/365*PXY
            RP_XY = 1 / (PXY * 365)

            HPHTRA_P.coords["time"] = pd.DatetimeIndex(HPHTRA_P.coords['time']).year
            HPHTRA_T.coords["time"] = pd.DatetimeIndex(HPHTRA_T.coords['time']).year
            HPHTRA_P.attrs["units"] = "%"
            HPHTRA_T.attrs["units"] = "%"
            a = xr.merge([HPHTRA_P.rename("HPHTRA_P"), HPHTRA_T.rename("HPHTRA_T")])
            HPHTRA1.append(a)
            RP_XY.name = "Return_Period"
            RP_XY.attrs["units"] = "yrs"
            Return_Period.append(RP_XY)
            del warm_and_wet, warm, wet, a
        xr.concat(HPHTRA1, dim='latitude').to_netcdf(self.Directory_CE)
        xr.concat(Return_Period, dim='latitude').to_netcdf(self.Directory_CE.split('.')[0] + "_Return_Period.nc")
        # xr.concat(HPHTRA1, dim='latitude').to_zarr(self.Directory_CE)
        # xr.concat(Return_Period, dim='latitude').to_zarr(self.Directory_CE.split('.')[0] + "_Return_Period.zarr")

    #Todo 复合事件年均平均持续天数

    def HighPrecipication_HighTemperatute_D4(self):
        print("高温强降雨复合事件重叠总天数")
        warm_dnd_wet_days_ByY=[]
        warm_dnd_wet_days_ByM = []
        for i in range(self.Rows):
            print(i)
            data1 = self.Files1[self.Var1][:,i,:]
            data2 = self.Files2[self.Var2][:,i,:]
            data1 = xclim.core.units.convert_units_to(data1, "degC")
            data2 = xclim.core.units.amount2rate(data2, out_units="mm/d")
            data2_wet = data2.where(data2 >= 1)
            das_per= percentile_doy(data1, window=5,per=float(self.Var1_Threshod)*100).sel(percentiles=float(self.Var1_Threshod)*100)
            pr_per = percentile_doy(data2_wet,window=30,per=float(self.Var2_Threshod)* 100).sel(percentiles=float(self.Var2_Threshod) * 100)
            # TODO method 1
            #warm_and_wet=xclim.indices.warm_and_wet_days(data1,data2,das_per,pr_per,freq='YS')
            # TODO method 2
            thresh1=resample_doy(das_per,data1)
            thresh2=resample_doy(pr_per,data2)
            das_per.rolling()
            tg75=data1>thresh1
            pr75=data2>thresh2
            warm_and_wet_ByY=np.logical_and(tg75, pr75).resample(time='YS').sum(dim="time")
            warm_and_wet_ByM = np.logical_and(tg75, pr75).resample(time='MS').sum(dim="time")
            warm_and_wet_ByY.coords["time"] = pd.DatetimeIndex(warm_and_wet_ByY.coords['time']).year
            warm_and_wet_ByM.coords["time"] = pd.DatetimeIndex(warm_and_wet_ByM.coords['time']).month
            warm_and_wet_ByY.name = "HPHTD2_Y"
            warm_and_wet_ByM.name = "HPHTD2_M"
            warm_and_wet_ByY.attrs["units"] = "days"
            warm_and_wet_ByY.attrs["units"] = "days"
            warm_dnd_wet_days_ByY.append(warm_and_wet_ByY)
            warm_dnd_wet_days_ByM.append(warm_and_wet_ByM)
            del warm_and_wet_ByY
            del warm_and_wet_ByM
        xr.concat(warm_dnd_wet_days_ByY, dim='latitude').load().to_netcdf(self.Directory_CE.split('.')[0] + "_ByY.nc")
        xr.concat(warm_dnd_wet_days_ByM, dim='latitude').load().to_netcdf(self.Directory_CE.split('.')[0] + "_ByM.nc")

            # xclim.indices.warm_and_wet_days
