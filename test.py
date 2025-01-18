# import h5py
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as n
import  xarray as xr
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
import time
from xclim.core.calendar import percentile_doy,resample_doy
from xclim.indices.generic import threshold_count,compare,to_agg_units
from xclim.core.units import Quantified
import xclim.indices.run_length as rl
import warnings
warnings.filterwarnings("ignore")

Files1=xr.open_mfdataset("D:\\Carpenter\\Projects\\ICES\\ExtractionCompoundEvents\\2m_temperature_Days\\2m_temperature_*.nc",concat_dim="time", combine="nested",data_vars='minimal', coords='minimal', compat='override')
Files2=xr.open_mfdataset("D:\\Carpenter\\Projects\\ICES\\ExtractionCompoundEvents\\total_precipitation_Days\\total_precipitation_*.nc",concat_dim="time", combine="nested",data_vars='minimal', coords='minimal', compat='override')


Var1='t2m'
Var2='tp'
import shutil
from pathlib import Path

def  del_file(path,a):
      for elm in Path(path).glob(a):
            print(elm)
            os.remove(elm) if elm.is_file() else shutil.rmtree(elm)


#TODO HPHTF1 绝对阈值 并且逐年保存数据--适用于内存小的电脑
def HighPrecipication_HighTemperatute_F1(Var1_Threshod,Var2_Threshod,Directory_CE1,Directory_CE2):
        print("高温强降雨复合事件重叠分布")
        Units = ['K', 'degC', 'm', 'mm']
        File_Type = np.array(['Temperature', 'Temperature', 'Precipitation', 'Precipitation'])
        for t in np.unique(pd.DatetimeIndex(Files1.coords['time']).year):
            print(t)
            data1 = Files1[Var1].sel(time=(Files1.time.dt.year==t)).load()
            data2 = Files2[Var2].sel(time=(Files2.time.dt.year==t)).load()
            if File_Type[np.isin(Units, Files1[Var1].units)] == "Temperature":
                data1 = xclim.core.units.convert_units_to(data1, "degC")
                data2 = xclim.core.units.amount2rate(data2, out_units="mm/d")
                Var2_Threshod = xclim.core.units.convert_units_to(float(Var2_Threshod), data2,
                                                                       context="hydro")
                Var1_Threshod = xclim.core.units.convert_units_to(float(Var1_Threshod), data1)
            else:
                data2 = xclim.core.units.convert_units_to(data2, "degC")
                data1 = xclim.core.units.amount2rate(data1, out_units="mm/d")
                Var1_Threshod = xclim.core.units.convert_units_to(float(Var1_Threshod), data1,
                                                                       context="hydro")
                Var2_Threshod = xclim.core.units.convert_units_to(float(Var2_Threshod), data2)
            pr75 = data1 > Var1_Threshod
            ht = data2 > Var2_Threshod

            pr75.to_netcdf(Directory_CE1+"t2m_"+str(int(Var1_Threshod))+"_"+str(int(Var2_Threshod))+"_"+str(t)+'.nc')
            ht.to_netcdf(Directory_CE2+"tp_"+str(int(Var1_Threshod))+"_"+str(int(Var2_Threshod))+"_"+str(t)+'.nc')

# HighPrecipication_HighTemperatute_F1(27,5,"E:\\Experiments\\BinaryMap\\t2m\\","E:\\Experiments\\BinaryMap\\pr\\")


def HighPrecipication_HighTemperatute_F3(Var1_Threshod,Var2_Threshod,Directory_CE1,Directory_CE2):
        print("高温强降雨复合事件重叠分布")
        fileName1=[]
        fileName2=[]
        # for t in np.unique(pd.DatetimeIndex(Files1[Var1].coords['time']).year):
        for t1 in np.unique(Files1.time.dt.year):
            file_A1=[]
            file_A2=[]
            for i in range(Files1[Var1].shape[1]):
                file_A1.append(Directory_CE1+"t2m_1_"+str(float(Var1_Threshod))+"_"+str(float(Var2_Threshod))+"_"+str(i)+"_"+str(t1)+'.nc')
                file_A2.append(Directory_CE2+"pr_1_"+str(float(Var1_Threshod))+"_"+str(float(Var2_Threshod))+"_"+str(i)+"_"+str(t1)+'.nc')
            fileName1.append(file_A1)
            fileName2.append(file_A2)
        for i in range(Files1[Var1].shape[1]):
            print(i)
            data1 = Files1[Var1].isel(latitude=i).load()
            data2 = Files2[Var2].isel(latitude=i).load()
            data1 = xclim.core.units.convert_units_to(data1, "degC")
            data2 = xclim.core.units.amount2rate(data2, out_units="mm/d")
            data2_wet = data2.where(data2 >= 1)
            das_per = percentile_doy(data1, window=5, per=float(Var1_Threshod) * 100).sel(
                    percentiles=float(Var1_Threshod) * 100)
            pr_per = percentile_doy(data2_wet, window=30, per=float(Var2_Threshod) * 100).sel(
                    percentiles=float(Var2_Threshod) * 100)
            das_per1 = resample_doy(das_per, data1)
            das_per2 = resample_doy(pr_per, data2)
            tg75 = data1 > das_per1.transpose('time','longitude')
            pr75 = data2 > das_per2.transpose('time','longitude')
            for t in np.unique(pd.DatetimeIndex(tg75.coords['time']).year):
                tg75.sel(time=(tg75.time.dt.year==t)).to_netcdf(Directory_CE1+"t2m_1_"+str(float(Var1_Threshod))+"_"+str(float(Var2_Threshod))+"_"+str(i)+"_"+str(t)+'.nc')
                pr75.sel(time=(pr75.time.dt.year==t)).to_netcdf(Directory_CE2+"pr_1_"+str(float(Var1_Threshod))+"_"+str(float(Var2_Threshod))+"_"+str(i)+"_"+str(t)+'.nc')
        for t in range(np.unique(Files1.time.dt.year).shape[0]):
            with open_mfdataset(fileName1[t],concat_dim="latitude",combine="nested",data_vars='minimal', coords='minimal', compat='override') as Filetp:
                 with open_mfdataset(fileName2[t],concat_dim="latitude", combine="nested",data_vars='minimal', coords='minimal', compat='override') as Filepr:
                    Filetp=Filetp.rename({"__xarray_dataarray_variable__": "t2m"})
                    Filepr=Filepr.rename({"__xarray_dataarray_variable__": "pr"})
                    Filetp.to_netcdf(Directory_CE1+"t2m_1_"+str(float(Var1_Threshod))+"_"+str(float(Var2_Threshod))+"_"+str(np.unique(Files1.time.dt.year)[t])+'.nc')
                    Filepr.to_netcdf(Directory_CE2+"pr_1_"+str(float(Var1_Threshod))+"_"+str(float(Var2_Threshod))+"_"+str(np.unique(Files1.time.dt.year)[t])+'.nc')
                    Filetp.close()
                    Filepr.close()
            del_file(Directory_CE1,"t2m_1_"+str(float(Var1_Threshod))+"_"+str(float(Var2_Threshod))+"_*_"+str(np.unique(Files1.time.dt.year)[t])+'.nc')
            del_file(Directory_CE2,"pr_1_"+str(float(Var1_Threshod))+"_"+str(float(Var2_Threshod))+"_*_"+str(np.unique(Files1.time.dt.year)[t])+'.nc')
            # del_file(Directory_CE1,"t2m_1_"+str(float(Var1_Threshod))+"_"+str(float(Var2_Threshod))+"_*_"+str(np.unique(Files1.time.dt.year)[t+5])+'.nc')
            # del_file(Directory_CE2,"pr_1_"+str(float(Var1_Threshod))+"_"+str(float(Var2_Threshod))+"_*_"+str(np.unique(Files1.time.dt.year)[t+5])+'.nc')

Value=[[0.8,0.7],[0.8,0.8],[0.8,0.9]]
for i in range(len(Value)):
        HighPrecipication_HighTemperatute_F3(Value[i][0],Value[i][1],"F:\\Experiments\\BinaryMap\\t2m_1\\","F:\\Experiments\\BinaryMap\\pr_1\\")

    #TODO HPHTF3 相对阈值1 并且逐年保存数据--适用于内存小的电脑
def HighPrecipication_HighTemperatute_F5(Var1_Threshod,Var2_Threshod,Directory_CE1,Directory_CE2):
        print("高温强降雨复合事件重叠分布")
        fileName1=[]
        fileName2=[]
        for t in np.unique(pd.DatetimeIndex(Files1[Var1].coords['time']).year):
            file_A1=[]
            file_A2=[]
            for i in range(Files1[Var1].shape[1]):
                file_A1.append(Directory_CE1+"t2m_2_"+str(float(Var1_Threshod))+"_"+str(float(Var2_Threshod))+"_"+str(i)+"_"+str(t)+'.nc')
                file_A2.append(Directory_CE2+"pr_2_"+str(float(Var1_Threshod))+"_"+str(float(Var2_Threshod))+"_"+str(i)+"_"+str(t)+'.nc')
            fileName1.append(file_A1)
            fileName2.append(file_A2)
        for i in range(Files1[Var1].shape[1]):
            print(i)
            data1 = Files1[Var1][:,i:i+1,:].load()
            data2 = Files2[Var2][:,i:i+1,:].load()
            data1 = xclim.core.units.convert_units_to(data1, "degC")
            data2 = xclim.core.units.amount2rate(data2, out_units="mm/d")
            data2_wet = data2.where(data2 >= 1)
            das_per1=data1.quantile([float(Var1_Threshod)], dim='time',keep_attrs=True).sel(quantile=Var1_Threshod).drop('quantile')
            das_per2 = data2_wet.quantile([float(Var2_Threshod)], dim='time',keep_attrs=True).sel(quantile=Var2_Threshod).drop('quantile')
            tg75 = data1 > das_per1
            pr75 = data2 > das_per2
            for t in np.unique(pd.DatetimeIndex(tg75.coords['time']).year):
                tg75.sel(time=(tg75.time.dt.year==t)).to_netcdf(Directory_CE1+"t2m_2_"+str(float(Var1_Threshod))+"_"+str(float(Var2_Threshod))+"_"+str(i)+"_"+str(t)+'.nc')
                pr75.sel(time=(pr75.time.dt.year==t)).to_netcdf(Directory_CE2+"pr_2_"+str(float(Var1_Threshod))+"_"+str(float(Var2_Threshod))+"_"+str(i)+"_"+str(t)+'.nc')
        for t in range(np.unique(Files1.time.dt.year).shape[0]):
            Filetp=open_mfdataset(fileName1[t],concat_dim="latitude",combine="nested",data_vars='minimal', coords='minimal', compat='override')
            Filepr=open_mfdataset(fileName2[t],concat_dim="latitude", combine="nested",data_vars='minimal', coords='minimal', compat='override')
            Filetp.to_netcdf(Directory_CE1+"t2m_2_"+str(float(Var1_Threshod))+"_"+str(float(Var2_Threshod))+"_"+str(np.unique(Files1.time.dt.year)[t])+'.nc')
            Filepr.to_netcdf(Directory_CE2+"pr_2_"+str(float(Var1_Threshod))+"_"+str(float(Var2_Threshod))+"_"+str(np.unique(Files1.time.dt.year)[t])+'.nc')
            Filepr.close()
            Filetp.close()
            del_file(Directory_CE1,"t2m_2_"+str(float(Var1_Threshod))+"_"+str(float(Var2_Threshod))+"_*_"+str(np.unique(Files1.time.dt.year)[t])+'.nc')
            del_file(Directory_CE2,"pr_2_"+str(float(Var1_Threshod))+"_"+str(float(Var2_Threshod))+"_*_"+str(np.unique(Files1.time.dt.year)[t])+'.nc')

# for i in range(len(Value)):
#     HighPrecipication_HighTemperatute_F5(Value[i][0],Value[i][1],"E:\\Experiments\\BinaryMap\\t2m_2\\","E:\\Experiments\\BinaryMap\\pr_2\\")
