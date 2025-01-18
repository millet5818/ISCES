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
from xarray import open_mfdataset,open_dataset
#import dask
from xclim import testing
from glob import glob
from xclim.core.calendar import percentile_doy,resample_doy
from xclim.indices.generic import threshold_count,compare,to_agg_units
from xclim.core.units import Quantified
import xclim.indices.run_length as rl


class CE_Index:
    def __init__(self,Files1,Files2,Threshold1,Threshold2,Directory_CE,var1,var2):
        self.Files1=Files1
        self.Files2=Files2
        self.Threshold1=Threshold1
        self.Threshold2=Threshold2
        self.Directory_CE=Directory_CE
        self.var1=var1
        self.var2=var2
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
                # warm_and_wet_ByY.attrs["units"] = "days"
                # warm_and_wet_ByM.attrs["units"] = "days"
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
            # warm_and_wet_ByY.attrs["units"] = "days"
            # warm_and_wet_ByM.attrs["units"] = "days"
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
            # out.attrs["units"] = "days"

            max_l_M = rl.resample_and_rl(cond,True,rl.longest_run,freq='MS')
            out_M = max_l_M.where(max_l_M >= 2, 0)
            out_M.coords["time"] = pd.DatetimeIndex(out_M.coords['time']).month
            out_M.name = "HPHTM_M"
            # out_M.attrs["units"] = "days"

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
            # out.attrs["units"] = "days"
            HPHTM_Y.append(out)

            max_l_M = rl.resample_and_rl(cond, True, rl.longest_run, freq='MS')
            out_M = max_l_M.where(max_l_M >= 2, 0)
            out_M.coords["time"] = pd.DatetimeIndex(out_M.coords['time']).month
            out_M.name = "HPHTM2_M"
            # out_M.attrs["units"] = "days"
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
        data1 = self.Files1[self.var1]
        data2 = self.Files2[self.var2]
        cond = np.logical_and(data1, data2)
        max_l = rl.resample_and_rl(cond, True, rl.longest_run, freq='YS')
        HPHTM_Y = max_l.where(max_l >= 2, 0)
        HPHTM_Y.coords["time"] = pd.DatetimeIndex(HPHTM_Y.coords['time']).year
        HPHTM_Y.name = "HPHTM3_Y"
        HPHTM_Y.attrs["units"] = "days"
        max_l_M = rl.resample_and_rl(cond, True, rl.longest_run, freq='MS')
        HPHTM_M = max_l_M.where(max_l_M >= 2, 0)
        HPHTM_M.coords["time"] = pd.DatetimeIndex(HPHTM_M.coords['time']).month
        HPHTM_M.name = "HPHTM3_M"
        HPHTM_M.attrs["units"] = "days"
        return HPHTM_Y, HPHTM_M
        # xr.concat(HPHTM_Y, dim='latitude').to_netcdf(self.Directory_CE+"HPHTM3_"+self.Threshold1+"_"+self.Threshold2 + "_ByY.nc")
        # xr.concat(HPHTM_M, dim='latitude').to_netcdf(self.Directory_CE+"HPHTM3_"+self.Threshold1+"_"+self.Threshold2 + "_ByM.nc")


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
        data1 = self.Files1[self.var1]
        data2 = self.Files2[self.var2]
        cond = np.logical_and(data1, data2)
        HPHTEN_Y = rl.resample_and_rl(cond, True, rl.windowed_run_events,window=1, freq='YS')
        HPHTEN_Y.coords["time"] = pd.DatetimeIndex(HPHTEN_Y.coords['time']).year
        HPHTEN_Y.name = "HPHTEN3_Y"
        HPHTEN_Y.attrs["units"] = ""

        HPHTEN_M = rl.resample_and_rl(cond, True, rl.windowed_run_events, window=1, freq='MS')
        HPHTEN_M.coords["time"] = pd.DatetimeIndex(HPHTEN_M.coords['time']).month
        HPHTEN_M.name = "HPHTEN3_M"
        HPHTEN_M.attrs["units"] = ""
        return HPHTEN_Y,HPHTEN_M

        # xr.concat(HPHTEN_Y, dim='latitude').to_netcdf(self.Directory_CE+"HPHTEN3_"+self.Threshold1+"_"+self.Threshold2 + "_ByY.nc")
        # xr.concat(HPHTEN_M, dim='latitude').to_netcdf(self.Directory_CE+"HPHTEN3_"+self.Threshold1+"_"+self.Threshold2 + "_ByM.nc")

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
        data1 = self.Files1[self.var1]
        data2 = self.Files2[self.var2]
        warm_and_wet = np.logical_and(data1, data2)
        warm_and_wet_year = warm_and_wet.resample(time='YS').sum(dim="time")
        warm = data1.resample(time='YS').sum(dim="time")
        wet = data2.resample(time='YS').sum(dim="time")
        HPHTRA_P = warm_and_wet_year / warm
        HPHTRA_T = warm_and_wet_year / wet
        HPHTRA_P.coords["time"] = pd.DatetimeIndex(HPHTRA_P.coords['time']).year
        HPHTRA_T.coords["time"] = pd.DatetimeIndex(HPHTRA_T.coords['time']).year
        HPHTRA_P.attrs["units"] = "%"
        HPHTRA_T.attrs["units"] = "%"
        # a = xr.merge([HPHTRA_P.rename("HPHTRA_P"), HPHTRA_T.rename("HPHTRA_T")])
        return HPHTRA_P,HPHTRA_T


    def HighPrecipication_HighTemperatute_Return(self):
            print("HPHTReturn")
            Return_Period = []
            for i in range(self.Files1.latitude.shape[0]):
                print(i)
                data1 = self.Files1[self.var1].isel(latitude=i).load()
                data2 = self.Files2[self.var2].isel(latitude=i).load()
                warm_and_wet = np.logical_and(data1, data2)
                warm_and_wet_allyear = warm_and_wet.sum(dim="time")
                PXY = warm_and_wet_allyear / warm_and_wet.time.dt.day.shape[0]
                # todo The Return period[RP with [RP] =years] is then Calculated as 1/365*PXY
                RP_XY = 1 / (PXY * 365)
                RP_XY[np.isinf(RP_XY)]=np.nan
                RP_XY.name = "Return_Period"
                RP_XY.attrs["units"] = "yrs"
                Return_Period.append(RP_XY)
            xr.concat(Return_Period, dim='latitude').to_netcdf(self.Directory_CE+"HPHTReturn_"+str(self.Threshold1)+"_"+str(self.Threshold2)+".nc")




# TODO 2
Threshold=[[0.9,0.7],[0.9,0.8],[0.9,0.9]]

for i in range(len(Threshold)):
        print(i)
        Files1=xr.open_mfdataset("F:\\Experiments\\BinaryMap\\t2m_1\\t2m_1_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_*.nc",concat_dim="time", combine="nested",data_vars='minimal', coords='minimal', compat='override')
        Files2=xr.open_mfdataset("F:\\Experiments\\BinaryMap\\pr_1\\pr_1_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_*.nc",concat_dim="time", combine="nested",data_vars='minimal', coords='minimal', compat='override')
        CE=CE_Index(Files1,Files2,str(Threshold[i][0]*10),str(Threshold[i][1]*10),"F:\\Experiments\\Relative Threshold2\\",'t2m','pr')
        Period_Return=CE.HighPrecipication_HighTemperatute_Return()


Directory_CE2="F:\\Experiments\\BinaryMap\\pr_1\\"
Directory_CE1="F:\\Experiments\\BinaryMap\\t2m_1\\"
fileName1=[]
fileName2=[]
for i in range(len(Threshold)):
        file_A1=[]
        file_A2=[]
        HPHTM_Y=[]
        HPHTM_M=[]
        HPHTEN_Y=[]
        HPHTEN_M=[]
        HPHTRA1=[]
        HPHTRA_P=[]
        HPHTRA_T=[]
        Return_Period=[]
        for t in np.arange(1941,2023,1):
            filename1=Directory_CE1+"t2m_1_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_"+str(t)+'.nc'
            filename2=Directory_CE2+"pr_1_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_"+str(t)+'.nc'
            with open_dataset(filename1) as ds1:
                with open_dataset(filename2) as ds2:
                           CE=CE_Index(ds1,ds2,str(Threshold[i][0]*10),str(Threshold[i][1]*10),"F:\\Experiments\\Relative Threshold2\\",'t2m','pr')
                           MY,MM=CE.HighPrecipication_HighTemperatute_MaxLength3()
                           HPHTM_Y.append(MY)
                           HPHTM_M.append(MM)
                           EY,EM=CE.HighPrecipication_HighTemperatute_EN3()
                           HPHTEN_Y.append(EY)
                           HPHTEN_M.append(EM)
                           RA_P,RA_T=CE.HighPrecipication_HighTemperatute_RA3()
                           HPHTRA_P.append(RA_P)
                           HPHTRA_T.append(RA_T)

        xr.concat(HPHTM_Y, dim='time').to_netcdf("F:\\Experiments\\Relative Threshold2\\HPHTM3_"+str(Threshold[i][0]*10)+"_"+str(Threshold[i][1]*10) + "_ByY.nc")
        xr.concat(HPHTM_M, dim='time').to_netcdf("F:\\Experiments\\Relative Threshold2\\HPHTM3_"+str(Threshold[i][0]*10)+"_"+str(Threshold[i][1]*10) + "_ByM.nc")
        xr.concat(HPHTEN_Y, dim='time').to_netcdf("F:\\Experiments\\Relative Threshold2\\HPHTEN3_"+str(Threshold[i][0]*10)+"_"+str(Threshold[i][1]*10) + "_ByY.nc")
        xr.concat(HPHTEN_M, dim='time').to_netcdf("F:\\Experiments\\Relative Threshold2\\HPHTEN3_"+str(Threshold[i][0]*10)+"_"+str(Threshold[i][1]*10)+ "_ByM.nc")
        xr.concat(HPHTRA_P, dim='time').to_netcdf("F:\\Experiments\\Relative Threshold2\\HPHTRA_P_"+str(Threshold[i][0]*10)+"_"+str(Threshold[i][1]*10) + "_ByY.nc")
        xr.concat(HPHTRA_T, dim='time').to_netcdf("F:\\Experiments\\Relative Threshold2\\HPHTRA_T_"+str(Threshold[i][0]*10)+"_"+str(Threshold[i][1]*10) + "_ByY.nc")


# # TODO 1
# Threshold=[[0.9,0.8],[0.9,0.9]]
# Directory_CE2="G:\\Experiments\\BinaryMap\\pr_2\\"
# Directory_CE1="G:\\Experiments\\BinaryMap\\t2m_2\\"
#
# for i in range(len(Threshold)):
#         file_A1=[]
#         file_A2=[]
#         HPHTM_Y=[]
#         HPHTM_M=[]
#         HPHTEN_Y=[]
#         HPHTEN_M=[]
#         HPHTRA1=[]
#         Return_Period=[]
#         for t in np.arange(1941,2023,1):
#             filename1=Directory_CE1+"t2m_2_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_"+str(t)+'.nc'
#             filename2=Directory_CE2+"pr_2_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_"+str(t)+'.nc'
#             with open_dataset(filename1) as ds1:
#                 with open_dataset(filename2) as ds2:
#                            CE=CE_Index(ds1,ds2,str(Threshold[i][0]*10),str(Threshold[i][1]*10),"G:\\Experiments\\Relative Threshold\\",'t2m','tp')
#                            MY,MM=CE.HighPrecipication_HighTemperatute_MaxLength3()
#                            HPHTM_Y.append(MY)
#                            HPHTM_M.append(MM)
#                            EY,EM=CE.HighPrecipication_HighTemperatute_EN3()
#                            HPHTEN_Y.append(EY)
#                            HPHTEN_M.append(EM)
#
#         xr.concat(HPHTM_Y, dim='time').to_netcdf("G:\\Experiments\\Relative Threshold\\HPHTM3_"+str(Threshold[i][0]*10)+"_"+str(Threshold[i][1]*10) + "_ByY.nc")
#         xr.concat(HPHTM_M, dim='time').to_netcdf("G:\\Experiments\\Relative Threshold\\HPHTM3_"+str(Threshold[i][0]*10)+"_"+str(Threshold[i][1]*10) + "_ByM.nc")
#         xr.concat(HPHTEN_Y, dim='time').to_netcdf("G:\\Experiments\\Relative Threshold\\HPHTEN3_"+str(Threshold[i][0]*10)+"_"+str(Threshold[i][1]*10) + "_ByY.nc")
#         xr.concat(HPHTEN_M, dim='time').to_netcdf("G:\\Experiments\\Relative Threshold\\HPHTEN3_"+str(Threshold[i][0]*10)+"_"+str(Threshold[i][1]*10)+ "_ByM.nc")


# TODO 2
# Threshold=[[0.7,0.9],[0.8,0.7],[0.8,0.8],[0.8,0.9],[0.9,0.7],[0.9,0.8],[0.9,0.9]]
# Directory_CE2="E:\\Experiments\\BinaryMap\\pr_2\\"
# Directory_CE1="E:\\Experiments\\BinaryMap\\t2m_2\\"
# for i in range(len(Threshold)):
#         file_A1=[]
#         file_A2=[]
#         HPHTRA_P=[]
#         HPHTRA_T=[]
#         for t in np.arange(1941,2023,1):
#             filename1=Directory_CE1+"t2m_2_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_"+str(t)+'.nc'
#             filename2=Directory_CE2+"pr_2_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_"+str(t)+'.nc'
#             with open_dataset(filename1) as ds1:
#                 with open_dataset(filename2) as ds2:
#                            CE=CE_Index(ds1,ds2,str(Threshold[i][0]*10),str(Threshold[i][1]*10),"G:\\Experiments\\Relative Threshold\\",'t2m','tp')
#                            RA_P,RA_T=CE.HighPrecipication_HighTemperatute_RA3()
#                            HPHTRA_P.append(RA_P)
#                            HPHTRA_T.append(RA_T)
#         xr.concat(HPHTRA_P, dim='time').to_netcdf("G:\\Experiments\\Relative Threshold\\HPHTRA_P_"+str(Threshold[i][0]*10)+"_"+str(Threshold[i][1]*10) + "_ByY.nc")
#         xr.concat(HPHTRA_T, dim='time').to_netcdf("G:\\Experiments\\Relative Threshold\\HPHTRA_T_"+str(Threshold[i][0]*10)+"_"+str(Threshold[i][1]*10) + "_ByY.nc")

        # Files1=xr.open_mfdataset("E:\\Experiments\\BinaryMap\\t2m_2\\t2m_2_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_*.nc",concat_dim="time", combine="nested",data_vars='minimal', coords='minimal', compat='override')
        # Files2=xr.open_mfdataset("E:\\Experiments\\BinaryMap\\pr_2\\pr_2_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_*.nc",concat_dim="time", combine="nested",data_vars='minimal', coords='minimal', compat='override')
        # CE=CE_Index(Files1,Files2,str(Threshold[i][0]*10),str(Threshold[i][1]*10),"G:\\Experiments\\Relative Threshold\\",'t2m','tp')
        # Period_Return=CE.HighPrecipication_HighTemperatute_Return()


Threshold=[[0.9,0.7],[0.9,0.8],[0.9,0.9]]
for i in range(len(Threshold)):
        with xr.open_mfdataset("E:\\Experiments\\BinaryMap\\t2m_2\\t2m_2_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_*.nc",concat_dim="time", combine="nested",data_vars='minimal', coords='minimal', compat='override') as Files1:
            with xr.open_mfdataset("E:\\Experiments\\BinaryMap\\pr_2\\pr_2_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_*.nc",concat_dim="time", combine="nested",data_vars='minimal', coords='minimal', compat='override') as Files2:
        # Files1=xr.open_mfdataset("E:\\Experiments\\BinaryMap\\t2m_2\\t2m_2_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_*.nc",concat_dim="time", combine="nested",data_vars='minimal', coords='minimal', compat='override')
        # Files2=xr.open_mfdataset("E:\\Experiments\\BinaryMap\\pr_2\\pr_2_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_*.nc",concat_dim="time", combine="nested",data_vars='minimal', coords='minimal', compat='override')
                CE=CE_Index(Files1,Files2,str(Threshold[i][0]*10),str(Threshold[i][1]*10),"G:\\Experiments\\Relative Threshold\\",'t2m','tp')
                Period_Return=CE.HighPrecipication_HighTemperatute_Return()


# Threshold=[[0.9,0.7],[0.9,0.8],[0.9,0.9]]
# Directory_CE2="E:\\Experiments\\BinaryMap\\pr_1\\"
# Directory_CE1="E:\\Experiments\\BinaryMap\\t2m_1\\"
# fileName1=[]
# fileName2=[]
# for i in range(len(Threshold)):
#         file_A1=[]
#         file_A2=[]
#         HPHTM_Y=[]
#         HPHTM_M=[]
#         HPHTEN_Y=[]
#         HPHTEN_M=[]
#         HPHTRA1=[]
#         HPHTRA_P=[]
#         HPHTRA_T=[]
#         for t in np.arange(1941,2023,1):
#             filename1=Directory_CE1+"t2m_1_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_"+str(t)+'.nc'
#             filename2=Directory_CE2+"pr_1_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_"+str(t)+'.nc'
#             with open_dataset(filename1) as ds1:
#                 with open_dataset(filename2) as ds2:
#                            CE=CE_Index(ds1,ds2,str(Threshold[i][0]*10),str(Threshold[i][1]*10),"G:\\Experiments\\Relative Threshold2\\",'t2m','pr')
#                            MY,MM=CE.HighPrecipication_HighTemperatute_MaxLength3()
#                            HPHTM_Y.append(MY)
#                            HPHTM_M.append(MM)
#                            EY,EM=CE.HighPrecipication_HighTemperatute_EN3()
#                            HPHTEN_Y.append(EY)
#                            HPHTEN_M.append(EM)
#                            RA_P,RA_T=CE.HighPrecipication_HighTemperatute_RA3()
#                            HPHTRA_P.append(RA_P)
#                            HPHTRA_T.append(RA_T)
#         xr.concat(HPHTM_Y, dim='time').to_netcdf("G:\\Experiments\\Relative Threshold2\\HPHTM3_"+str(Threshold[i][0]*10)+"_"+str(Threshold[i][1]*10) + "_ByY.nc")
#         xr.concat(HPHTM_M, dim='time').to_netcdf("G:\\Experiments\\Relative Threshold2\\HPHTM3_"+str(Threshold[i][0]*10)+"_"+str(Threshold[i][1]*10) + "_ByM.nc")
#         xr.concat(HPHTEN_Y, dim='time').to_netcdf("G:\\Experiments\\Relative Threshold2\\HPHTEN3_"+str(Threshold[i][0]*10)+"_"+str(Threshold[i][1]*10) + "_ByY.nc")
#         xr.concat(HPHTEN_M, dim='time').to_netcdf("G:\\Experiments\\Relative Threshold2\\HPHTEN3_"+str(Threshold[i][0]*10)+"_"+str(Threshold[i][1]*10)+ "_ByM.nc")
#         xr.concat(HPHTRA_P, dim='time').to_netcdf("G:\\Experiments\\Relative Threshold2\\HPHTRA_P_"+str(Threshold[i][0]*10)+"_"+str(Threshold[i][1]*10) + "_ByY.nc")
#         xr.concat(HPHTRA_T, dim='time').to_netcdf("G:\\Experiments\\Relative Threshold2\\HPHTRA_T_"+str(Threshold[i][0]*10)+"_"+str(Threshold[i][1]*10) + "_ByY.nc")
#
#         Files1=xr.open_mfdataset("E:\\Experiments\\BinaryMap\\t2m_1\\t2m_1_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_*.nc",concat_dim="time", combine="nested",data_vars='minimal', coords='minimal', compat='override')
#         Files2=xr.open_mfdataset("E:\\Experiments\\BinaryMap\\pr_1\\pr_1_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_*.nc",concat_dim="time", combine="nested",data_vars='minimal', coords='minimal', compat='override')
#         CE=CE_Index(Files1,Files2,str(Threshold[i][0]*10),str(Threshold[i][1]*10),"G:\\Experiments\\Relative Threshold2\\",'t2m','pr')
#         Period_Return=CE.HighPrecipication_HighTemperatute_Return()
