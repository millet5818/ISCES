"""
To plot analysis results about times-domain, spatial-domain, change detection, 联合概率，etc
"""
import os
import xarray as xr
import numpy as np
import salem
import matplotlib.pyplot  as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import cmaps
from cartopy.util import add_cyclic_point
# from corr_2d_ttest import *
# from corr_sig import *
from cartopy.io.shapereader import Reader
from matplotlib import colors
import matplotlib
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import jenkspy
mpl.rcParams["font.sans-serif"] = ["Times New Roman"] # TODO  Windows_Functions
mpl.rcParams['font.size'] = 16  # 设置全局字体大小
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# matplotlib.use('Agg')
import matplotlib.style as mplstyle
mplstyle.use('fast')

# # 函数封装，方便调用
def add_geo_ticks(ax, proj, extent, lat_span = 5, lon_span = 5):
    '''
        在图中添加刻度信息
        Version 1.0
    '''
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 刻度格式
    offset = min(lon_span, lat_span)
    xticks = np.arange(extent[0], extent[1] + offset, lon_span)  # 创建x轴刻度
    yticks = np.arange(extent[2], extent[3] + offset, lat_span)  # 创建y轴刻度
    # 添加刻度
    ax.set_xticks(xticks, crs=proj)
    ax.set_yticks(yticks, crs=proj)
    # 设置刻度格式为经纬度格式
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    # 设置画图范围
    ax.set_extent(extent)

class PlotAnalysis:
    def __init__(self,file,shapefile,variable):
        self.shapefile=shapefile
        self.file=file
        self.variable=variable
        self.Files1 = xr.open_dataset(self.file)
        # self.Files2 = xr.open_mfdataset(self.shapefile, concat_dim="time", combine="nested", data_vars='minimal',coords='minimal', compat='override')
        # self.Files1 = xr.open_zarr(self.file)
    def ReadFile(self):
        print("读取文件")

    def ZarrToNC(self):
        print("Zarr转换NC数据")
        self.Files1.load().to_netcdf(self.file.split('.')[0]+'.nc')
    """
    Mask:遮罩 
    Zone_Mask:分区文件
    Statistical_Method:统计方法(sum、mean)
    统计区段：Year,Month,Season,Day
    Export_Csv_File:导出文件名
    
    """
    def Mask_ZHOU(self,x,Geometry):

        return x.salem.roi(geometry=Geometry)
    def Mask_ZHOU1(self,x,Geometry):

        return x.sum(dim='time').salem.roi(geometry=Geometry)
    def Time_Domain_Analysis_Year(self):
        print("事件域分析")
        "D1:"
        OutputFile=self.file.split('.')[0]
        Sum_Year=self.Files1[self.variable].sum(dim=["latitude", "longitude"])
        Mean_Year=self.Files1[self.variable].mean(dim=["latitude", "longitude"])*100
        np.savetxt(OutputFile+'_Sum_Year.csv', np.array([Sum_Year.time,Sum_Year]).T,delimiter=",",fmt='%f')
        np.savetxt(OutputFile+'_Mean_Year.csv', np.array([Mean_Year.time,Mean_Year]).T,delimiter=",",fmt='%f')

        Header=['Year']
        Sum_Year_Clip=[]
        Mean_Year_Clip=[]
        gdf = gpd.read_file(self.shapefile)
        for i in range(len(gdf)):
            print(i)
            Name=gdf.CONTINENT[i]
            Header.append(Name)
            Geometry=gdf.geometry[i]
            Data=self.Files1[self.variable].groupby(self.Files1.time).map(lambda x: self.Mask_ZHOU(x,Geometry))
            Sum_Year_clip = Data.sum(dim=["latitude", "longitude"])
            Sum_Year_clip.name=str(Name)
            Sum_Year_Clip.append(Sum_Year_clip)
            Mean_Year_clip = Data.mean(dim=["latitude", "longitude"])*100
            Mean_Year_clip.name=str(Name)
            Mean_Year_Clip.append(Mean_Year_clip)
        Sum_Year_Data=xr.merge(Sum_Year_Clip)
        Mean_Year_Data=xr.merge(Mean_Year_Clip)
        names = [n for n in Header]
        np.savetxt(OutputFile+"_Sum_Year_Division.csv", np.concatenate((np.expand_dims(Sum_Year_Data.time.values,axis=1),Sum_Year_Data.to_array().values.T),axis=1),delimiter=",",fmt='%f',header=','.join(names))
        np.savetxt(OutputFile+"_Mean_Year_Division.csv", np.concatenate((np.expand_dims(Mean_Year_Data.time.values,axis=1),Mean_Year_Data.to_array().values.T),axis=1),delimiter=",",fmt='%f',header=','.join(names))

    def Time_Domain_Analysis_Month(self):
        print("事件域分析")
        "D1:"
        OutputFile=self.file.split('.')[0]
        Sum_Year=self.Files1[self.variable].groupby(self.Files1[self.variable].time).sum().sum(dim=["latitude", "longitude"])
        Mean_Year=self.Files1[self.variable].groupby(self.Files1[self.variable].time).sum().mean(dim=["latitude", "longitude"])*100
        np.savetxt(OutputFile+'_Sum_Month.csv', np.array([Sum_Year.time,Sum_Year]).T,delimiter=",",fmt='%f')
        np.savetxt(OutputFile+'_Mean_Month.csv', np.array([Mean_Year.time,Mean_Year]).T,delimiter=",",fmt='%f')

        Header=['Month']
        Sum_Year_Clip=[]
        Mean_Year_Clip=[]
        gdf = gpd.read_file(self.shapefile)
        for i in range(len(gdf)):
            print(i)
            Name=gdf.CONTINENT[i]
            Header.append(Name)
            Geometry=gdf.geometry[i]
            Data=self.Files1[self.variable].groupby(self.Files1.time).map(lambda x: self.Mask_ZHOU1(x,Geometry))
            Sum_Year_clip = Data.sum(dim=["latitude", "longitude"])
            Sum_Year_clip.name=str(Name)
            Sum_Year_Clip.append(Sum_Year_clip)
            Mean_Year_clip = Data.mean(dim=["latitude", "longitude"])*100
            Mean_Year_clip.name=str(Name)
            Mean_Year_Clip.append(Mean_Year_clip)
        Sum_Year_Data=xr.merge(Sum_Year_Clip)
        Mean_Year_Data=xr.merge(Mean_Year_Clip)
        names = [n for n in Header]
        np.savetxt(OutputFile+"_Sum_Month_Division.csv", np.concatenate((np.expand_dims(Sum_Year_Data.time.values,axis=1),Sum_Year_Data.to_array().values.T),axis=1),delimiter=",",fmt='%f',header=','.join(names))
        np.savetxt(OutputFile+"_Mean_Month_Division.csv", np.concatenate((np.expand_dims(Mean_Year_Data.time.values,axis=1),Mean_Year_Data.to_array().values.T),axis=1),delimiter=",",fmt='%f',header=','.join(names))

    def Time_Domain_Analysis_Year1(self):
        print("事件域分析")
        "D3:"
        OutputFile=self.file.split('.')[0]
        Sum_Year=self.Files1[self.variable].dt.days.sum(dim=["latitude", "longitude"])
        Mean_Year=self.Files1[self.variable].dt.days.mean(dim=["latitude", "longitude"])*100
        np.savetxt(OutputFile+'_Sum_Year.csv', np.array([Sum_Year.time,Sum_Year.sel(quantile=0)]).T,delimiter=",",fmt='%f')
        np.savetxt(OutputFile+'_Mean_Year.csv', np.array([Mean_Year.time,Mean_Year.sel(quantile=0)]).T,delimiter=",",fmt='%f')

        Header=['Year']
        Sum_Year_Clip=[]
        Mean_Year_Clip=[]
        gdf = gpd.read_file(self.shapefile)
        for i in range(len(gdf)):
            print(i)
            Name=gdf.CONTINENT[i]
            Header.append(Name)
            Geometry=gdf.geometry[i]
            Data=self.Files1[self.variable].dt.days.groupby(self.Files1.time).map(lambda x: self.Mask_ZHOU(x.sel(quantile=0),Geometry))
            Sum_Year_clip = Data.sum(dim=["latitude", "longitude"])
            Sum_Year_clip.name=str(Name)
            Sum_Year_Clip.append(Sum_Year_clip)
            Mean_Year_clip = Data.mean(dim=["latitude", "longitude"])*100
            Mean_Year_clip.name=str(Name)
            Mean_Year_Clip.append(Mean_Year_clip)
        Sum_Year_Data=xr.merge(Sum_Year_Clip)
        Mean_Year_Data=xr.merge(Mean_Year_Clip)
        names = [n for n in Header]
        np.savetxt(OutputFile+"_Sum_Year_Division.csv", np.concatenate((np.expand_dims(Sum_Year_Data.time.values,axis=1),Sum_Year_Data.to_array().values.T),axis=1),delimiter=",",fmt='%f',header=','.join(names))
        np.savetxt(OutputFile+"_Mean_Year_Division.csv", np.concatenate((np.expand_dims(Mean_Year_Data.time.values,axis=1),Mean_Year_Data.to_array().values.T),axis=1),delimiter=",",fmt='%f',header=','.join(names))

    def Time_Domain_Analysis_Month1(self):
        print("事件域分析")
        "D3:"
        OutputFile=self.file.split('.')[0]
        Data=self.Files1[self.variable].groupby(self.Files1[self.variable].time).sum().dt.days
        Sum_Year=Data.sum(dim=["latitude", "longitude"])
        Mean_Year=Data.mean(dim=["latitude", "longitude"])*100
        np.savetxt(OutputFile+'_Sum_Month.csv', np.array([Sum_Year.time,Sum_Year.sel(quantile=0)]).T,delimiter=",",fmt='%f')
        np.savetxt(OutputFile+'_Mean_Month.csv', np.array([Mean_Year.time,Mean_Year.sel(quantile=0)]).T,delimiter=",",fmt='%f')

        Header=['Month']
        Sum_Year_Clip=[]
        Mean_Year_Clip=[]
        gdf = gpd.read_file(self.shapefile)
        Data_ALL=self.Files1[self.variable].dt.days.groupby(self.Files1.time)
        for i in range(len(gdf)):
            print(i)
            Name=gdf.CONTINENT[i]
            Header.append(Name)
            Geometry=gdf.geometry[i]
            Data=Data_ALL.map(lambda x: self.Mask_ZHOU1(x.sel(quantile=0),Geometry))
            Sum_Year_clip = Data.sum(dim=["latitude", "longitude"])
            Sum_Year_clip.name=str(Name)
            Sum_Year_Clip.append(Sum_Year_clip)
            Mean_Year_clip = Data.mean(dim=["latitude", "longitude"])*100
            Mean_Year_clip.name=str(Name)
            Mean_Year_Clip.append(Mean_Year_clip)
        Sum_Year_Data=xr.merge(Sum_Year_Clip)
        Mean_Year_Data=xr.merge(Mean_Year_Clip)
        names = [n for n in Header]
        np.savetxt(OutputFile+"_Sum_Month_Division.csv", np.concatenate((np.expand_dims(Sum_Year_Data.time.values,axis=1),Sum_Year_Data.to_array().values.T),axis=1),delimiter=",",fmt='%f',header=','.join(names))
        np.savetxt(OutputFile+"_Mean_Month_Division.csv", np.concatenate((np.expand_dims(Mean_Year_Data.time.values,axis=1),Mean_Year_Data.to_array().values.T),axis=1),delimiter=",",fmt='%f',header=','.join(names))

    def Time_Domain_Analysis_Year2(self):
        print("事件域分析")
        "M1:"
        OutputFile=self.file.split('.')[0]
        Sum_Year=self.Files1[self.variable].dt.days.sum(dim=["latitude", "longitude"])
        Mean_Year=self.Files1[self.variable].dt.days.mean(dim=["latitude", "longitude"])*100
        np.savetxt(OutputFile+'_Sum_Year.csv', np.array([Sum_Year.time,Sum_Year]).T,delimiter=",",fmt='%f')
        np.savetxt(OutputFile+'_Mean_Year.csv', np.array([Mean_Year.time,Mean_Year]).T,delimiter=",",fmt='%f')

        Header=['Year']
        Sum_Year_Clip=[]
        Mean_Year_Clip=[]
        gdf = gpd.read_file(self.shapefile)
        for i in range(len(gdf)):
            print(i)
            Name=gdf.CONTINENT[i]
            Header.append(Name)
            Geometry=gdf.geometry[i]
            Data=self.Files1[self.variable].dt.days.groupby(self.Files1.time).map(lambda x: self.Mask_ZHOU(x,Geometry))
            Sum_Year_clip = Data.sum(dim=["latitude", "longitude"])
            Sum_Year_clip.name=str(Name)
            Sum_Year_Clip.append(Sum_Year_clip)
            Mean_Year_clip = Data.mean(dim=["latitude", "longitude"])*100
            Mean_Year_clip.name=str(Name)
            Mean_Year_Clip.append(Mean_Year_clip)
        Sum_Year_Data=xr.merge(Sum_Year_Clip)
        Mean_Year_Data=xr.merge(Mean_Year_Clip)
        names = [n for n in Header]
        np.savetxt(OutputFile+"_Sum_Year_Division.csv", np.concatenate((np.expand_dims(Sum_Year_Data.time.values,axis=1),Sum_Year_Data.to_array().values.T),axis=1),delimiter=",",fmt='%f',header=','.join(names))
        np.savetxt(OutputFile+"_Mean_Year_Division.csv", np.concatenate((np.expand_dims(Mean_Year_Data.time.values,axis=1),Mean_Year_Data.to_array().values.T),axis=1),delimiter=",",fmt='%f',header=','.join(names))

    def Time_Domain_Analysis_Month2(self):
        print("事件域分析")
        "D3:"
        OutputFile=self.file.split('.')[0]
        Data=self.Files1[self.variable].groupby(self.Files1[self.variable].time).sum().dt.days
        Sum_Year=Data.sum(dim=["latitude", "longitude"])
        Mean_Year=Data.mean(dim=["latitude", "longitude"])*100
        np.savetxt(OutputFile+'_Sum_Month.csv', np.array([Sum_Year.time,Sum_Year]).T,delimiter=",",fmt='%f')
        np.savetxt(OutputFile+'_Mean_Month.csv', np.array([Mean_Year.time,Mean_Year]).T,delimiter=",",fmt='%f')

        Header=['Month']
        Sum_Year_Clip=[]
        Mean_Year_Clip=[]
        gdf = gpd.read_file(self.shapefile)
        Data_All=self.Files1[self.variable].groupby(self.Files1.time).sum().dt.days
        for i in range(len(gdf)):
            print(i)
            Name=gdf.CONTINENT[i]
            Header.append(Name)
            Geometry=gdf.geometry[i]
            Data=Data_All.groupby('time').map(lambda x: self.Mask_ZHOU(x,Geometry))
            Sum_Year_clip = Data.sum(dim=["latitude", "longitude"])
            Sum_Year_clip.name=str(Name)
            Sum_Year_Clip.append(Sum_Year_clip)
            Mean_Year_clip = Data.mean(dim=["latitude", "longitude"])*100
            Mean_Year_clip.name=str(Name)
            Mean_Year_Clip.append(Mean_Year_clip)
        Sum_Year_Data=xr.merge(Sum_Year_Clip)
        Mean_Year_Data=xr.merge(Mean_Year_Clip)
        names = [n for n in Header]
        np.savetxt(OutputFile+"_Sum_Month_Division.csv", np.concatenate((np.expand_dims(Sum_Year_Data.time.values,axis=1),Sum_Year_Data.to_array().values.T),axis=1),delimiter=",",fmt='%f',header=','.join(names))
        np.savetxt(OutputFile+"_Mean_Month_Division.csv", np.concatenate((np.expand_dims(Mean_Year_Data.time.values,axis=1),Mean_Year_Data.to_array().values.T),axis=1),delimiter=",",fmt='%f',header=','.join(names))


    def Time_Domain_Analysis_Year3(self):
        print("事件域分析")
        "HPHTEN3_Y:"
        OutputFile=self.file.split('.')[0]
        Sum_Year=self.Files1[self.variable].sum(dim=["latitude", "longitude"])
        Mean_Year=self.Files1[self.variable].mean(dim=["latitude", "longitude"])*100
        np.savetxt(OutputFile+'_Sum_Year.csv', np.array([Sum_Year.time,Sum_Year]).T,delimiter=",",fmt='%f')
        np.savetxt(OutputFile+'_Mean_Year.csv', np.array([Mean_Year.time,Mean_Year]).T,delimiter=",",fmt='%f')
        Header=['Year']
        Sum_Year_Clip=[]
        Mean_Year_Clip=[]
        gdf = gpd.read_file(self.shapefile)
        for i in range(len(gdf)):
            print(i)
            Name=gdf.CONTINENT[i]
            Header.append(Name)
            Geometry=gdf.geometry[i]
            Data=self.Files1[self.variable].groupby(self.Files1.time).map(lambda x: self.Mask_ZHOU(x,Geometry))
            Sum_Year_clip = Data.sum(dim=["latitude", "longitude"])
            Sum_Year_clip.name=str(Name)
            Sum_Year_Clip.append(Sum_Year_clip)
            Mean_Year_clip = Data.mean(dim=["latitude", "longitude"])*100
            Mean_Year_clip.name=str(Name)
            Mean_Year_Clip.append(Mean_Year_clip)
        Sum_Year_Data=xr.merge(Sum_Year_Clip)
        Mean_Year_Data=xr.merge(Mean_Year_Clip)
        names = [n for n in Header]
        np.savetxt(OutputFile+"_Sum_Year_Division.csv", np.concatenate((np.expand_dims(Sum_Year_Data.time.values,axis=1),Sum_Year_Data.to_array().values.T),axis=1),delimiter=",",fmt='%f',header=','.join(names))
        np.savetxt(OutputFile+"_Mean_Year_Division.csv", np.concatenate((np.expand_dims(Mean_Year_Data.time.values,axis=1),Mean_Year_Data.to_array().values.T),axis=1),delimiter=",",fmt='%f',header=','.join(names))


    def Time_Domain_Analysis_Month3(self):
        print("事件域分析")
        "HPHTEN3_M:"
        OutputFile=self.file.split('.')[0]
        Data=self.Files1[self.variable].groupby(self.Files1[self.variable].time).sum()
        Sum_Year=Data.sum(dim=["latitude", "longitude"])
        Mean_Year=Data.mean(dim=["latitude", "longitude"])*100
        np.savetxt(OutputFile+'_Sum_Month.csv', np.array([Sum_Year.time,Sum_Year]).T,delimiter=",",fmt='%f')
        np.savetxt(OutputFile+'_Mean_Month.csv', np.array([Mean_Year.time,Mean_Year]).T,delimiter=",",fmt='%f')
        Header=['Month']
        Sum_Year_Clip=[]
        Mean_Year_Clip=[]
        gdf = gpd.read_file(self.shapefile)
        Data_All=self.Files1[self.variable].groupby(self.Files1.time).sum()
        for i in range(len(gdf)):
            print(i)
            Name=gdf.CONTINENT[i]
            Header.append(Name)
            Geometry=gdf.geometry[i]
            Data=Data_All.groupby('time').map(lambda x: self.Mask_ZHOU(x,Geometry))
            Sum_Year_clip = Data.sum(dim=["latitude", "longitude"])
            Sum_Year_clip.name=str(Name)
            Sum_Year_Clip.append(Sum_Year_clip)
            Mean_Year_clip = Data.mean(dim=["latitude", "longitude"])*100
            Mean_Year_clip.name=str(Name)
            Mean_Year_Clip.append(Mean_Year_clip)
        Sum_Year_Data=xr.merge(Sum_Year_Clip)
        Mean_Year_Data=xr.merge(Mean_Year_Clip)
        names = [n for n in Header]
        np.savetxt(OutputFile+"_Sum_Month_Division.csv", np.concatenate((np.expand_dims(Sum_Year_Data.time.values,axis=1),Sum_Year_Data.to_array().values.T),axis=1),delimiter=",",fmt='%f',header=','.join(names))
        np.savetxt(OutputFile+"_Mean_Month_Division.csv", np.concatenate((np.expand_dims(Mean_Year_Data.time.values,axis=1),Mean_Year_Data.to_array().values.T),axis=1),delimiter=",",fmt='%f',header=','.join(names))


    def Time_Domain_Analysis_Year4(self):
        print("事件域分析")
        "M1:"
        OutputFile=self.file.split('.')[0]
        Sum_Year=self.Files1[self.variable].dt.days.max(dim=["latitude", "longitude"])
        np.savetxt(OutputFile+'_Max_Year.csv', np.array([Sum_Year.time,Sum_Year]).T,delimiter=",",fmt='%f')

        Header=['Year']
        Sum_Year_Clip=[]
        gdf = gpd.read_file(self.shapefile)
        for i in range(len(gdf)):
            print(i)
            Name=gdf.CONTINENT[i]
            Header.append(Name)
            Geometry=gdf.geometry[i]
            Data=self.Files1[self.variable].dt.days.groupby(self.Files1.time).map(lambda x: self.Mask_ZHOU(x,Geometry))
            Sum_Year_clip = Data.max(dim=["latitude", "longitude"])
            Sum_Year_clip.name=str(Name)
            Sum_Year_Clip.append(Sum_Year_clip)
        Sum_Year_Data=xr.merge(Sum_Year_Clip)
        names = [n for n in Header]
        np.savetxt(OutputFile+"_Max_Year_Division.csv", np.concatenate((np.expand_dims(Sum_Year_Data.time.values,axis=1),Sum_Year_Data.to_array().values.T),axis=1),delimiter=",",fmt='%f',header=','.join(names))

    def Time_Domain_Analysis_Month4(self):
        print("事件域分析")
        "D3:"
        OutputFile=self.file.split('.')[0]
        Data=self.Files1[self.variable].groupby(self.Files1[self.variable].time).max().dt.days
        Sum_Year=Data.max(dim=["latitude", "longitude"])
        np.savetxt(OutputFile+'_Max_Month.csv', np.array([Sum_Year.time,Sum_Year]).T,delimiter=",",fmt='%f')

        Header=['Month']
        Sum_Year_Clip=[]
        gdf = gpd.read_file(self.shapefile)
        Data_All=self.Files1[self.variable].groupby(self.Files1.time).max().dt.days
        for i in range(len(gdf)):
            print(i)
            Name=gdf.CONTINENT[i]
            Header.append(Name)
            Geometry=gdf.geometry[i]
            Data=Data_All.groupby('time').map(lambda x: self.Mask_ZHOU(x,Geometry))
            Sum_Year_clip = Data.max(dim=["latitude", "longitude"])
            Sum_Year_clip.name=str(Name)
            Sum_Year_Clip.append(Sum_Year_clip)
        Sum_Year_Data=xr.merge(Sum_Year_Clip)
        names = [n for n in Header]
        np.savetxt(OutputFile+"_Max_Month_Division.csv", np.concatenate((np.expand_dims(Sum_Year_Data.time.values,axis=1),Sum_Year_Data.to_array().values.T),axis=1),delimiter=",",fmt='%f',header=','.join(names))


    def Spatial_Domain_Analysis(self):
        print("空间域分析")
        file1=xr.open_dataset('E:/Experiments/Relative Threshold/HPHTReturn_7_7.nc')
        file2=xr.open_dataset('E:/Experiments/Relative Threshold/HPHTReturn_7_8.nc')
        file3=xr.open_dataset('E:/Experiments/Relative Threshold/HPHTReturn_7_9.nc')
        file4=xr.open_dataset('E:/Experiments/Relative Threshold/HPHTReturn_8_7.nc')
        file5=xr.open_dataset('E:/Experiments/Relative Threshold/HPHTReturn_8_8.nc')
        file6=xr.open_dataset('E:/Experiments/Relative Threshold/HPHTReturn_8_9.nc')
        file7=xr.open_dataset('E:/Experiments/Relative Threshold/HPHTReturn_9_7.nc')
        file8=xr.open_dataset('E:/Experiments/Relative Threshold/HPHTReturn_9_8.nc')
        file9=xr.open_dataset('E:/Experiments/Relative Threshold/HPHTReturn_9_9.nc')
        file10=xr.open_dataset('E:/Experiments/Abosolute Threshold/HPHTRA1_27_5_Return_Period.nc')

        ds1=file1
        lon_name = 'longitude'  # whatever name is in the data
        ds1['_longitude_adjusted'] = xr.where(ds1[lon_name] > 180,ds1[lon_name] - 360,ds1[lon_name])
        ds1 = (ds1.swap_dims({lon_name: '_longitude_adjusted'}).sel(**{'_longitude_adjusted': sorted(ds1._longitude_adjusted)}).drop(lon_name))
        ds1 = ds1.rename({'_longitude_adjusted': lon_name})

        ds2=file2
        lon_name = 'longitude'  # whatever name is in the data
        ds2['_longitude_adjusted'] = xr.where(ds2[lon_name] > 180,ds2[lon_name] - 360,ds2[lon_name])
        ds2 = (ds2.swap_dims({lon_name: '_longitude_adjusted'}).sel(**{'_longitude_adjusted': sorted(ds2._longitude_adjusted)}).drop(lon_name))
        ds2 = ds2.rename({'_longitude_adjusted': lon_name})

        ds3=file3
        lon_name = 'longitude'  # whatever name is in the data
        ds3['_longitude_adjusted'] = xr.where(ds3[lon_name] > 180,ds3[lon_name] - 360,ds3[lon_name])
        ds3 = (ds3.swap_dims({lon_name: '_longitude_adjusted'}).sel(**{'_longitude_adjusted': sorted(ds3._longitude_adjusted)}).drop(lon_name))
        ds3 = ds3.rename({'_longitude_adjusted': lon_name})

        ds4=file4
        lon_name = 'longitude'  # whatever name is in the data
        ds4['_longitude_adjusted'] = xr.where(ds4[lon_name] > 180,ds4[lon_name] - 360,ds4[lon_name])
        ds4 = (ds4.swap_dims({lon_name: '_longitude_adjusted'}).sel(**{'_longitude_adjusted': sorted(ds4._longitude_adjusted)}).drop(lon_name))
        ds4 = ds4.rename({'_longitude_adjusted': lon_name})

        ds5=file5
        lon_name = 'longitude'  # whatever name is in the data
        ds5['_longitude_adjusted'] = xr.where(ds5[lon_name] > 180,ds5[lon_name] - 360,ds5[lon_name])
        ds5 = (ds5.swap_dims({lon_name: '_longitude_adjusted'}).sel(**{'_longitude_adjusted': sorted(ds5._longitude_adjusted)}).drop(lon_name))
        ds5 = ds5.rename({'_longitude_adjusted': lon_name})

        ds6=file6
        lon_name = 'longitude'  # whatever name is in the data
        ds6['_longitude_adjusted'] = xr.where(ds6[lon_name] > 180,ds6[lon_name] - 360,ds6[lon_name])
        ds6 = (ds6.swap_dims({lon_name: '_longitude_adjusted'}).sel(**{'_longitude_adjusted': sorted(ds6._longitude_adjusted)}).drop(lon_name))
        ds6 = ds6.rename({'_longitude_adjusted': lon_name})

        ds7=file7
        lon_name = 'longitude'  # whatever name is in the data
        ds7['_longitude_adjusted'] = xr.where(ds7[lon_name] > 180,ds7[lon_name] - 360,ds7[lon_name])
        ds7 = (ds7.swap_dims({lon_name: '_longitude_adjusted'}).sel(**{'_longitude_adjusted': sorted(ds7._longitude_adjusted)}).drop(lon_name))
        ds7 = ds7.rename({'_longitude_adjusted': lon_name})

        ds8=file8
        lon_name = 'longitude'  # whatever name is in the data
        ds8['_longitude_adjusted'] = xr.where(ds8[lon_name] > 180,ds8[lon_name] - 360,ds8[lon_name])
        ds8 = (ds8.swap_dims({lon_name: '_longitude_adjusted'}).sel(**{'_longitude_adjusted': sorted(ds8._longitude_adjusted)}).drop(lon_name))
        ds8 = ds8.rename({'_longitude_adjusted': lon_name})

        ds9=file9
        lon_name = 'longitude'  # whatever name is in the data
        ds9['_longitude_adjusted'] = xr.where(ds9[lon_name] > 180,ds9[lon_name] - 360,ds9[lon_name])
        ds9 = (ds9.swap_dims({lon_name: '_longitude_adjusted'}).sel(**{'_longitude_adjusted': sorted(ds9._longitude_adjusted)}).drop(lon_name))
        ds9 = ds9.rename({'_longitude_adjusted': lon_name})

        ds10=file10
        lon_name = 'longitude'  # whatever name is in the data
        ds10['_longitude_adjusted'] = xr.where(ds10[lon_name] > 180,ds10[lon_name] - 360,ds10[lon_name])
        ds10 = (ds10.swap_dims({lon_name: '_longitude_adjusted'}).sel(**{'_longitude_adjusted': sorted(ds10._longitude_adjusted)}).drop(lon_name))
        ds10 = ds10.rename({'_longitude_adjusted': lon_name})


        longitude=ds1.variables['longitude'][:]
        latitude=ds1.variables['latitude'][:]

        # slow
        gdf = gpd.read_file(self.shapefile)
        maskregion1 = ds1.salem.roi(shape=gdf)
        data1=maskregion1[self.variable].values
        maskregion2 = ds2.salem.roi(shape=gdf)
        data2=maskregion2[self.variable].values
        maskregion3 = ds3.salem.roi(shape=gdf)
        data3=maskregion3[self.variable].values
        maskregion4 = ds4.salem.roi(shape=gdf)
        data4=maskregion4[self.variable].values
        maskregion5 = ds5.salem.roi(shape=gdf)
        data5=maskregion5[self.variable].values
        maskregion6 = ds6.salem.roi(shape=gdf)
        data6=maskregion6[self.variable].values
        maskregion7 = ds7.salem.roi(shape=gdf)
        data7=maskregion7[self.variable].values
        maskregion8 = ds8.salem.roi(shape=gdf)
        data8=maskregion8[self.variable].values
        maskregion9 = ds9.salem.roi(shape=gdf)
        data9=maskregion9[self.variable].values
        maskregion10 = ds10.salem.roi(shape=gdf)
        data10=maskregion10[self.variable].values


        vmin = min(np.nanmin(data1), np.nanmin(data2), np.nanmin(data3), np.nanmin(data4), np.nanmin(data5), np.nanmin(data6),np.nanmin(data7),np.nanmin(data8),np.nanmin(data9),np.nanmin(data10))
        vmax = max(np.nanmax(data1), np.nanmax(data2), np.nanmax(data3), np.nanmax(data4), np.nanmax(data5), np.nanmax(data6),np.nanmax(data7),np.nanmax(data8),np.nanmax(data9),np.nanmax(data10))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)


        fig = plt.figure(figsize=(16, 9))
        proj = ccrs.PlateCarree()
        datadata=[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10]
        for i in range(10):
            print(i)
            ax1 = fig.add_subplot(111, projection=ccrs.Robinson())
            ax1.gridlines(draw_labels=True, alpha=0.2,dms=True, x_inline=False, y_inline=False)
            ax1.coastlines( color='k', linewidth=0.2, alpha=0.4, linestyle='--')
            ax1.add_feature(cfeature.LAND,facecolor='gray',alpha=0.2,linestyle='-',lw=0.25)
            china_map = cfeature.ShapelyFeature(shpreader.Reader(self.shapefile).geometries(), proj, edgecolor='k', facecolor='none')
            ax1.add_feature(china_map,alpha=0.4, linewidth=0.1)
            qq1=ax1.contourf(longitude, latitude, datadata[i], extend = 'both',norm=norm,transform=ccrs.PlateCarree(),levels = 30,cmap='rainbow')
        # fig.colorbar(qq,label="Return period (yrs)", orientation="horizontal",shrink=0.5,pad=0.06)
            fig.colorbar(qq1,ax=[ax1],label="Return period (yrs)", orientation="horizontal",shrink=0.5,pad=0.06,ticks=np.linspace(vmin, vmax, 5))
        # fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
            plt.savefig(str(i)+"_Return period.png",dpi=800)
            plt.show()


        # ax1 = fig.add_subplot(331, projection=ccrs.Robinson())
        # ax2 = fig.add_subplot(332, projection=ccrs.Robinson())
        # ax3 = fig.add_subplot(333, projection=ccrs.Robinson())
        # ax4 = fig.add_subplot(334, projection=ccrs.Robinson())
        # ax5 = fig.add_subplot(335, projection=ccrs.Robinson())
        # ax6 = fig.add_subplot(336, projection=ccrs.Robinson())
        # ax7 = fig.add_subplot(337, projection=ccrs.Robinson())
        # ax8 = fig.add_subplot(338, projection=ccrs.Robinson())
        # ax9 = fig.add_subplot(339, projection=ccrs.Robinson())
        #
        # # slow
        # ax1.gridlines(draw_labels=True, alpha=0.2,dms=True, x_inline=False, y_inline=False)
        # ax1.coastlines( color='k', linewidth=0.2, alpha=0.4, linestyle='--')
        # ax1.add_feature(cfeature.LAND,facecolor='gray',alpha=0.2,linestyle='-',lw=0.25)
        # china_map = cfeature.ShapelyFeature(shpreader.Reader(self.shapefile).geometries(), proj, edgecolor='k', facecolor='none')
        # ax1.add_feature(china_map,alpha=0.4, linewidth=0.1)
        #
        # ax2.gridlines(draw_labels=True, alpha=0.2,dms=True, x_inline=False, y_inline=False)
        # ax2.coastlines( color='k', linewidth=0.2, alpha=0.4, linestyle='--')
        # ax2.add_feature(cfeature.LAND,facecolor='gray',alpha=0.2,linestyle='-',lw=0.25)
        # ax2.add_feature(china_map,alpha=0.4, linewidth=0.1)
        #
        # ax3.gridlines(draw_labels=True, alpha=0.2,dms=True, x_inline=False, y_inline=False)
        # ax3.coastlines( color='k', linewidth=0.2, alpha=0.4, linestyle='--')
        # ax3.add_feature(cfeature.LAND,facecolor='gray',alpha=0.2,linestyle='-',lw=0.25)
        # ax3.add_feature(china_map,alpha=0.4, linewidth=0.1)
        #
        # ax4.gridlines(draw_labels=True, alpha=0.2,dms=True, x_inline=False, y_inline=False)
        # ax4.coastlines( color='k', linewidth=0.2, alpha=0.4, linestyle='--')
        # ax4.add_feature(cfeature.LAND,facecolor='gray',alpha=0.2,linestyle='-',lw=0.25)
        # ax4.add_feature(china_map,alpha=0.4, linewidth=0.1)
        #
        # ax5.gridlines(draw_labels=True, alpha=0.2,dms=True, x_inline=False, y_inline=False)
        # ax5.coastlines( color='k', linewidth=0.2, alpha=0.4, linestyle='--')
        # ax5.add_feature(cfeature.LAND,facecolor='gray',alpha=0.2,linestyle='-',lw=0.25)
        # ax5.add_feature(china_map,alpha=0.4, linewidth=0.1)
        #
        # ax6.gridlines(draw_labels=True, alpha=0.2,dms=True, x_inline=False, y_inline=False)
        # ax6.coastlines( color='k', linewidth=0.2, alpha=0.4, linestyle='--')
        # ax6.add_feature(cfeature.LAND,facecolor='gray',alpha=0.2,linestyle='-',lw=0.25)
        # ax6.add_feature(china_map,alpha=0.4, linewidth=0.1)
        #
        # ax7.gridlines(draw_labels=True, alpha=0.2,dms=True, x_inline=False, y_inline=False)
        # ax7.coastlines( color='k', linewidth=0.2, alpha=0.4, linestyle='--')
        # ax7.add_feature(cfeature.LAND,facecolor='gray',alpha=0.2,linestyle='-',lw=0.25)
        # ax7.add_feature(china_map,alpha=0.4, linewidth=0.1)
        #
        # ax8.gridlines(draw_labels=True, alpha=0.2,dms=True, x_inline=False, y_inline=False)
        # ax8.coastlines( color='k', linewidth=0.2, alpha=0.4, linestyle='--')
        # ax8.add_feature(cfeature.LAND,facecolor='gray',alpha=0.2,linestyle='-',lw=0.25)
        # ax8.add_feature(china_map,alpha=0.4, linewidth=0.1)
        #
        # ax9.gridlines(draw_labels=True, alpha=0.2,dms=True, x_inline=False, y_inline=False)
        # ax9.coastlines( color='k', linewidth=0.2, alpha=0.4, linestyle='--')
        # ax9.add_feature(cfeature.LAND,facecolor='gray',alpha=0.2,linestyle='-',lw=0.25)
        # ax9.add_feature(china_map,alpha=0.4, linewidth=0.1)
        #
        # # viridis slow
        # qq1=ax1.contourf(longitude, latitude, data1, extend = 'both',norm=norm,transform=ccrs.PlateCarree(),levels = 18,cmap='rainbow')# viridis
        # qq2=ax2.contourf(longitude, latitude, data2, extend = 'both',norm=norm,transform=ccrs.PlateCarree(),levels = 18,cmap='rainbow')# viridis
        # qq3=ax3.contourf(longitude, latitude, data3, extend = 'both',norm=norm,transform=ccrs.PlateCarree(),levels = 18,cmap='rainbow')# viridis
        # qq4=ax4.contourf(longitude, latitude, data4, extend = 'both',norm=norm,transform=ccrs.PlateCarree(),levels = 18,cmap='rainbow')# viridis
        # qq5=ax5.contourf(longitude, latitude, data5, extend = 'both',norm=norm,transform=ccrs.PlateCarree(),levels = 18,cmap='rainbow')# viridis
        # qq6=ax6.contourf(longitude, latitude, data6, extend = 'both',norm=norm,transform=ccrs.PlateCarree(),levels = 18,cmap='rainbow')# viridis
        # qq7=ax7.contourf(longitude, latitude, data7, extend = 'both',norm=norm,transform=ccrs.PlateCarree(),levels = 18,cmap='rainbow')# viridis
        # qq8=ax8.contourf(longitude, latitude, data8, extend = 'both',norm=norm,transform=ccrs.PlateCarree(),levels = 18,cmap='rainbow')# viridis
        # qq9=ax9.contourf(longitude, latitude, data9, extend = 'both',norm=norm,transform=ccrs.PlateCarree(),levels = 18,cmap='rainbow')# viridis
        #
        # # ax.set_title("High temperature and Heavy precipitation",fontsize=15)
        # # fig.colorbar(qq,label="Return period (yrs)", orientation="horizontal",shrink=0.5,pad=0.06)
        # fig.colorbar(qq1,ax=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9],label="Return period (yrs)", orientation="horizontal",shrink=0.5,pad=0.1)
        # # fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
        # plt.savefig("Return period.png",dpi=800)
        # plt.show()


    def Frequency_Domain_Analysis(self):
        print("频率域分析")
    def CubeConv3D(self,window,method):
        time=self.Files1.HPHTD1.time.shape[0]
        longitude=self.Files1.HPHTD1.longitude.shape[0]
        latitude=self.Files1.HPHTD1.latitude.shape[0]
        for t in range(time):
            print(t)
            for lo in range(longitude):
                for la in range(latitude):
                    self.Files1.HPHTD1[la,t,lo]=self.Files1.HPHTD1[la:la+window+1,t:t+window+1,lo:lo+window+1].max(dim=['time','latitude','longitude'])
        return self.Files1.rolling()

    def SpatialCorrelation(self):
        print("空间相关性分析")
        air = self.Files1.air[372:,:,:]
        lat0 = self.Files1.variables['lat'][:]
        lon0 = self.Files1.variables['lon'][:]
        # North America surface air temperature
        # field1 (time,lat,lon)
        air_NA = air.sel(lat=slice(90,-90),lon=slice(0,360))
        print(air_NA)
        # Los Angeles surface air temperature
        # field2 (time)
        air_LA = air.sel(lat=36,lon=100,method='nearest')
        print(air_LA)
        #Get lat and lon of air_NA
        # lat,lon should be consistent with the size of field1
        lat = air_NA.lat
        lon = air_NA.lon
        # longitude=self.Files1.variables['longitude'][:]
        # latitude=self.Files1.variables['latitude'][:]
        # data1 = xclim.core.units.convert_units_to(self.Files1['t2m'], "degC")
        # data2 = xclim.core.units.amount2rate(self.Files2['tp'], out_units="mm/d")
        options = SET(nsim=1000, method='isospectral', alpha=0.05)
        corr,latex,lonex = corr_2d_ttest(air_NA,air_LA,lat,lon,options,1)
        region=[-180, 180, -90, 90]
        proj=ccrs.PlateCarree()
        fig=plt.figure(figsize=(16,9),dpi=600)
        ax = plt.axes(projection  = proj)
        ax.set_extent(region, crs = proj)
        ax.coastlines(lw=0.4)
        ax.set_global()
        ax.stock_img()
        cycle_corr,cycle_lon=add_cyclic_point(corr,coord=lon)
        cycle_LON,cycle_LAT=np.meshgrid(cycle_lon,lat)
        # clevs=np.linspace(-1,1,0.2)
        cs=ax.contourf(cycle_LON, cycle_LAT,cycle_corr,np.arange(-1,1,0.2),cmap=cmaps.NCV_bright,extend='both')
        cbar=plt.colorbar(cs,shrink=0.75,orientation='vertical',extend='both',pad=0.015,aspect=30) #orientation='horizontal'
        cbar.set_label('(℃)')
        ax.set_xticks(np.arange(region[0], region[1] + 1, 60), crs = proj)
        ax.set_yticks(np.arange(region[-2], region[-1] + 1, 30), crs = proj)
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        font3={'family':'SimHei','size':12,'color':'k'}
        plt.title("全球海表温度分布图",fontdict=font3)
        plt.ylabel("纬度",fontdict=font3)
        plt.xlabel("经度",fontdict=font3)
        sig1=ax.scatter(lonex,latex,marker='.',s=1,c='k',alpha=0.6,transform=ccrs.PlateCarree())
        # plt.savefig('F:/Rpython/lp36/plot45.3.png',dpi=800,bbox_inches='tight',pad_inches=0)
        plt.show()

    def SpatialCorrelation2_Two(self):
        print("计算两变量的空间相关性")

    def SpatialTrends(self):
        print("单变量在时间序列上的空间趋势分析")

        ds=self.Files1
        lon_name = 'longitude'  # whatever name is in the data
        ds['_longitude_adjusted'] = xr.where(ds[lon_name] > 180,ds[lon_name] - 360,ds[lon_name])
        ds = (ds.swap_dims({lon_name: '_longitude_adjusted'}).sel(**{'_longitude_adjusted': sorted(ds._longitude_adjusted)}).drop(lon_name))
        ds = ds.rename({'_longitude_adjusted': lon_name})
        longitude=ds.variables['longitude'][:]
        latitude=ds.variables['latitude'][:]
        # ds1=ds.HPHTD1.isel(time=1)

        # todo 计算P值
        # slope=np.zeros((self.Files1.HPHTD1.latitude.shape[0],self.Files1.HPHTD1.longitude.shape[0]))
        # p_value=np.zeros((self.Files1.HPHTD1.latitude.shape[0],self.Files1.HPHTD1.longitude.shape[0]))
        #
        # for i in range (0,self.Files1.HPHTD1.latitude.shape[0]):
        #     print(i)
        #     for j in range (0,self.Files1.HPHTD1.longitude.shape[0]):
        #         slope[i,j],intercept,r_value,p_value[i,j],std_err=stats.linregress(np.arange(1941,2023,1),ds.HPHTD1[i,:,j].values)

        slope=np.load('slope.npy')
        p_value=np.load('p_value.npy')

        ds1_slope = ds.HPHTD1.isel(time=1)
        ds1_slope.values=slope

        ds1_pvalue = ds.HPHTD1.isel(time=1)
        ds1_pvalue.values = p_value

        gdf = gpd.read_file(r'../Data/continent.shp')
        mask_slope = ds1_slope.salem.roi(shape=gdf)
        mask_pvalue = ds1_pvalue.salem.roi(shape=gdf)

        fig=plt.figure(dpi=200,figsize=(16,9))
        proj=ccrs.PlateCarree()
        ax = plt.axes(projection = proj)
        region=[-180, 180, -90, 90]
        ax.set_extent(region, crs = proj)
        ax.gridlines(draw_labels=True, alpha=0.2,dms=True, x_inline=False, y_inline=False)
        ax.coastlines( color='k', linewidth=0.2, alpha=0.4, linestyle='--')
        ax.add_geometries(Reader(r'../Data/continent.shp').geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.2)
        ax.add_feature(cfeature.LAND,facecolor='gray',alpha=0.2,linestyle='-',lw=0.25)

        c11=ax.contourf(longitude,latitude,mask_slope.values,np.arange(-1,1,0.1),extend='both',levels=18,transform=ccrs.PlateCarree(),cmap='gist_rainbow',zorder=0)
        plt.colorbar(c11,label="Return period (yrs)", orientation="horizontal",shrink=0.5,pad=0.06)
        # 显著性打点
        sig1=ax.contourf(longitude,latitude,mask_pvalue,[np.min(p_value),0.5,np.max(p_value)],hatches=['..',None],colors="none",transform=ccrs.PlateCarree(),zorder=0)
        # font3={'family':'SimHei','size':12,'color':'k'}
        # ax.set_title('CMIP6_ssp126情景模式降雨和气温的空间相关性分析及显著性检验',fontdict=font3)
        # plt.savefig('../Data/plot94.1.png',dpi=200,bbox_inches='tight',pad_inches=0)
        plt.show()

    def Spatial_Events_Number(self):
        OutputFile=self.file.split('.')[0]
        print("复合事件发生灾害总和以及年均平均持续天数")
        file1=xr.open_dataset('F:/Experiments/Relative Threshold/HPHTEN3_7_7_ByY.nc')["HPHTEN3_Y"].sum(dim='time')

        common_levels2 = np.around(jenkspy.jenks_breaks(np.unique(file1.values.reshape(-1)),12),1)

        Mean_Year=file1
        ds1=Mean_Year
        lon_name = 'longitude'  # whatever name is in the data
        ds1['_longitude_adjusted'] = xr.where(ds1[lon_name] > 180,ds1[lon_name] - 360,ds1[lon_name])
        ds1 = (ds1.swap_dims({lon_name: '_longitude_adjusted'}).sel(**{'_longitude_adjusted': sorted(ds1._longitude_adjusted)}).drop(lon_name))
        ds1 = ds1.rename({'_longitude_adjusted': lon_name})


        longitude=ds1.longitude
        latitude=ds1.latitude
        gdf = gpd.read_file(self.shapefile)
        maskregion1 = ds1.salem.roi(shape=gdf)
        data1=maskregion1.values

        fig1 = plt.figure(figsize=(16, 9))
        proj = ccrs.PlateCarree()

        ax1 = fig1.add_subplot(111, projection=ccrs.Robinson())
        ax1.gridlines(draw_labels=True, alpha=0.2,dms=True, x_inline=False, y_inline=False)
        ax1.coastlines(color='k', linewidth=0.2, alpha=0.4, linestyle='--')
        ax1.add_feature(cfeature.LAND,facecolor='gray',alpha=0.2,linestyle='-',lw=0.25)
        china_map = cfeature.ShapelyFeature(shpreader.Reader(self.shapefile).geometries(), proj, edgecolor='k', facecolor='none')
        ax1.add_feature(china_map,alpha=0.4, linewidth=0.1)
        cmap = plt.get_cmap('viridis')
        # Colors = cmap.colors.insert(0,matplotlib.colors.to_rgb('#B0C4DE'))
        # newcolor=ListedColormap(Colors)
        Colors = (
        '#B0C4DE', '#6495ED', '#4169E1', '#0000FF', '#00BFFF', '#00FA9A', '#008B8B','#007500', '#006400', '#FFB5B5',
        '#FF0000', '#CE0000')
        qq1=ax1.contourf(longitude, latitude, data1, extend = 'both',levels=common_levels2,transform=ccrs.PlateCarree(),colors=Colors)# viridis
        fig1.colorbar(qq1,ax=[ax1],label="Total Events", ticks=common_levels2,orientation="horizontal",shrink=0.5,pad=0.06)
        # # fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
        plt.savefig(OutputFile+"Sum.png",dpi=800)
        plt.show()
    def Spatial_Days_Number(self):
        OutputFile=self.file.split('.')[0]
        print("复合事件发生灾害总和以及年均平均持续天数")
        file1=xr.open_dataset('G:/Experiments/Relative Threshold/HPHTD3_7_7_ByY.nc')["HPHTD3_Y"].dt.days.mean(dim='time').sel(quantile=0)
        file2=xr.open_dataset('G:/Experiments/Relative Threshold/HPHTD3_7_8_ByY.nc')["HPHTD3_Y"].dt.days.mean(dim='time').sel(quantile=0)
        file3=xr.open_dataset('G:/Experiments/Relative Threshold/HPHTD3_7_9_ByY.nc')["HPHTD3_Y"].dt.days.mean(dim='time').sel(quantile=0)
        file4=xr.open_dataset('G:/Experiments/Relative Threshold/HPHTD3_8_7_ByY.nc')["HPHTD3_Y"].dt.days.mean(dim='time').sel(quantile=0)
        file5=xr.open_dataset('G:/Experiments/Relative Threshold/HPHTD3_8_8_ByY.nc')["HPHTD3_Y"].dt.days.mean(dim='time').sel(quantile=0)
        file6=xr.open_dataset('G:/Experiments/Relative Threshold/HPHTD3_8_9_ByY.nc')["HPHTD3_Y"].dt.days.mean(dim='time').sel(quantile=0)
        file7=xr.open_dataset('G:/Experiments/Relative Threshold/HPHTD3_9_7_ByY.nc')["HPHTD3_Y"].dt.days.mean(dim='time').sel(quantile=0)
        file8=xr.open_dataset('G:/Experiments/Relative Threshold/HPHTD3_9_8_ByY.nc')["HPHTD3_Y"].dt.days.mean(dim='time').sel(quantile=0)
        file9=xr.open_dataset('G:/Experiments/Relative Threshold/HPHTD3_9_9_ByY.nc')["HPHTD3_Y"].dt.days.mean(dim='time').sel(quantile=0)
        file10=xr.open_dataset('G:/Experiments/Abosolute Threshold/HPHTD1_27_5_ByY.nc')["HPHTD1_Y"].mean(dim='time')

        vmin = min(np.nanmin(file1), np.nanmin(file2), np.nanmin(file3), np.nanmin(file4), np.nanmin(file5), np.nanmin(file6),np.nanmin(file7),np.nanmin(file8),np.nanmin(file9),np.nanmin(file10))
        vmax = max(np.nanmax(file1), np.nanmax(file2), np.nanmax(file3), np.nanmax(file4), np.nanmax(file5), np.nanmax(file6),np.nanmax(file7),np.nanmax(file8),np.nanmax(file9),np.nanmax(file10))

        norm1 = colors.Normalize(vmin=vmin, vmax=vmax)


        Mean_Year=self.Files1[self.variable].mean(dim='time')
        ds1=Mean_Year
        lon_name = 'longitude'  # whatever name is in the data
        ds1['_longitude_adjusted'] = xr.where(ds1[lon_name] > 180,ds1[lon_name] - 360,ds1[lon_name])
        ds1 = (ds1.swap_dims({lon_name: '_longitude_adjusted'}).sel(**{'_longitude_adjusted': sorted(ds1._longitude_adjusted)}).drop(lon_name))
        ds1 = ds1.rename({'_longitude_adjusted': lon_name})


        longitude=ds1.longitude
        latitude=ds1.latitude
        gdf = gpd.read_file(self.shapefile)
        maskregion1 = ds1.salem.roi(shape=gdf)
        data1=maskregion1.values



        fig1 = plt.figure(figsize=(16, 9))
        proj = ccrs.PlateCarree()

        ax1 = fig1.add_subplot(111, projection=ccrs.Robinson())
        ax1.gridlines(draw_labels=True, alpha=0.2,dms=True, x_inline=False, y_inline=False)
        ax1.coastlines(color='k', linewidth=0.2, alpha=0.4, linestyle='--')
        ax1.add_feature(cfeature.LAND,facecolor='gray',alpha=0.2,linestyle='-',lw=0.25)
        china_map = cfeature.ShapelyFeature(shpreader.Reader(self.shapefile).geometries(), proj, edgecolor='k', facecolor='none')
        ax1.add_feature(china_map,alpha=0.4, linewidth=0.1)

        # ax2 = fig2.add_subplot(111, projection=ccrs.Robinson())
        # ax2.coastlines( color='k', linewidth=0.2, alpha=0.4, linestyle='--')
        # ax2.add_feature(cfeature.LAND,facecolor='gray',alpha=0.2,linestyle='-',lw=0.25)
        # ax2.add_feature(china_map,alpha=0.4, linewidth=0.1)

        qq1=ax1.contourf(longitude, latitude, data1, extend = 'both',norm=norm1,transform=ccrs.PlateCarree(),levels = 18,cmap='rainbow')# viridis
        # qq2=ax2.contourf(longitude, latitude, data2, extend = 'both',norm=norm2,transform=ccrs.PlateCarree(),levels = 18,cmap='viridis')# viridis

        # fig1.colorbar(qq1,ax=[ax1],label="Total Events", orientation="horizontal",shrink=0.5,pad=0.06)
        fig1.colorbar(qq1,ax=[ax1],label="Total days/Year", orientation="horizontal",shrink=0.5,pad=0.06)
        # # fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
        plt.savefig(OutputFile+"MeanDays.png",dpi=800)
        # plt.show()
    def Spatial_Days_Number2(self):
        OutputFile=self.file.split('.')[0]
        print("复合事件发生灾害总和以及年均平均持续天数")
        file1=xr.open_dataset('F:/Experiments/Relative Threshold/HPHTD3_7_7_ByY.nc')["HPHTD3_Y"].dt.days.mean(dim='time').sel(quantile=0)

        vmin = np.nanmin(file1)
        vmax = np.nanmax(file1)

        # norm1 = colors.Normalize(vmin=vmin, vmax=vmax)
        # common_levels = np.linspace(vmin,vmax, 12)
        common_levels2 = np.around(jenkspy.jenks_breaks(np.unique(file1.values.reshape(-1)),12),1)
        ds1=file1
        lon_name = 'longitude'  # whatever name is in the data
        ds1['_longitude_adjusted'] = xr.where(ds1[lon_name] > 180,ds1[lon_name] - 360,ds1[lon_name])
        ds1 = (ds1.swap_dims({lon_name: '_longitude_adjusted'}).sel(**{'_longitude_adjusted': sorted(ds1._longitude_adjusted)}).drop(lon_name))
        ds1 = ds1.rename({'_longitude_adjusted': lon_name})


        longitude=ds1.longitude
        latitude=ds1.latitude
        gdf = gpd.read_file(self.shapefile)
        maskregion1 = ds1.salem.roi(shape=gdf)
        data1=maskregion1.values

        fig1 = plt.figure(figsize=(16, 9))

        proj = ccrs.PlateCarree()

        ax1 = fig1.add_subplot(111, projection=ccrs.Robinson())
        ax1.gridlines(draw_labels=True, alpha=0.2,dms=True, x_inline=False, y_inline=False)
        ax1.coastlines(color='k', linewidth=0.2, alpha=0.4, linestyle='--')
        ax1.add_feature(cfeature.LAND,facecolor='gray',alpha=0.2,linestyle='-',lw=0.25)
        china_map = cfeature.ShapelyFeature(shpreader.Reader(self.shapefile).geometries(), proj, edgecolor='k', facecolor='none')
        ax1.add_feature(china_map,alpha=0.4, linewidth=0.1)
        Colors = (
        '#B0C4DE', '#6495ED', '#4169E1', '#0000FF', '#00BFFF', '#00FA9A', '#008B8B','#007500', '#006400', '#FFB5B5',
        '#FF0000', '#CE0000')
        # qq1=ax1.contourf(longitude, latitude, data1, extend = 'both',norm=norm1,transform=ccrs.PlateCarree(),levels = 18,cmap='viridis')# viridis
        qq1=ax1.contourf(longitude, latitude, data1, extend='both',levels=common_levels2,transform=ccrs.PlateCarree(),colors=Colors)

        # colors = ['black', 'red', 'yellow', 'cyan']
        # bounds = [0, 1, 2, 3]
        # cmap = matplotlib.colors.ListedColormap(colors, N=4)
        # norm = matplotlib.colors.BoundaryNorm([-1, 0.1, 1.1, 2.1, 3.1], cmap.N)

        # fig1.colorbar(qq1,ax=[ax1],label="Total Events", orientation="horizontal",shrink=0.5,pad=0.06)
        cb=fig1.colorbar(qq1,ax=[ax1],label="Total days/Year", extend='both',ticks=common_levels2, orientation="horizontal",shrink=0.5,pad=0.08)
        # cb.ax.tick_params(labelsize=16)
        # cb.ax.yaxis.set_major_locator(MultipleLocator(4))
        # cb.ax.yaxis.set_minor_locator(MultipleLocator(2))

        plt.savefig(OutputFile+"Mean.png",dpi=800)
        plt.show()



pt = PlotAnalysis("F:/Experiments/Relative Threshold/HPHTEN3_7_7_ByY.nc","../Data/dazhou.shp",'HPHTEN3_Y')
pt.Spatial_Events_Number()



# Threshold=[[7,7],[7,8],[7,9],[8,7],[8,8],[8,9],[9,7],[9,8],[9,9]]
# for i in range(len(Threshold)):
#     print(i)
#     pt = PlotAnalysis("G:\\Experiments\\Relative Threshold\\HPHTD3_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_ByY.nc","../Data/dazhou.shp",'HPHTD3_Y')



# pt = PlotAnalysis("D:\\Carpenter\\Projects\\ICES\\ExtractionCompoundEvents\\dasd_ByYZZR.nc","../Data/HPHTD1_27_5.nc")
# pt = PlotAnalysis("D:\\Carpenter\\Projects\\ICES\\ExtractionCompoundEvents\\dasd_ByYZZR.nc","../Data/HPHTD1_27_5.nc")
# pt = PlotAnalysis("E:\Experiments\Relative Threshold\\HPHTD3_9_9_ByY.nc","E:\Experiments\Relative Threshold\\HPHTD3_9_9_ByY.nc")

# pt = PlotAnalysis("../Data/air.mon.mean.nc","../Data/air.mon.mean.nc")


















# # todo 1
# pt = PlotAnalysis("E:\\Experiments\\Abosolute Threshold\\HPHTD1_27_5_ByY.nc","../Data/dazhou.shp",'HPHTD1_Y')
# pt.Time_Domain_Analysis_Year()
# # todo 2
# pt = PlotAnalysis("E:\\Experiments\\Abosolute Threshold\\HPHTD1_27_5_ByM.nc","../Data/dazhou.shp",'HPHTD1_M')
# pt.Time_Domain_Analysis_Month()
# # todo 3
# Threshold=[[7,7],[7,8],[7,9],[8,7],[8,8],[8,9],[9,7],[9,8],[9,9]]
# Filename=[]
# for i in range(len(Threshold)):
#     pt = PlotAnalysis("E:\\Experiments\\Relative Threshold\\HPHTD3_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_ByY.nc","../Data/dazhou.shp",'HPHTD3_Y')
#     pt.Time_Domain_Analysis_Year1()
# # todo 4
# Threshold=[[7,7],[7,8],[7,9],[8,7],[8,8],[8,9],[9,7],[9,8],[9,9]]
# Filename=[]
# for i in range(len(Threshold)):
#     pt = PlotAnalysis("E:\\Experiments\\Relative Threshold\\HPHTD3_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_ByM.nc","../Data/dazhou.shp",'HPHTD3_M')
#     pt.Time_Domain_Analysis_Month1()
#
# # todo 5
# pt = PlotAnalysis("E:\\Experiments\\Abosolute Threshold\\HPHTEN1_27_5_ByY.nc","../Data/dazhou.shp",'HPHTEN1_Y')
# pt.Time_Domain_Analysis_Year()
#
# # todo 6
# pt = PlotAnalysis("E:\\Experiments\\Abosolute Threshold\\HPHTEN1_27_5_ByM.nc","../Data/dazhou.shp",'HPHTEN1_M')
# pt.Time_Domain_Analysis_Month()

# todo 7
# pt = PlotAnalysis("E:\\Experiments\\Abosolute Threshold\\HPHTM1_27_5_ByY.nc","../Data/dazhou.shp",'HPHTM_Y')
# pt.Time_Domain_Analysis_Year2()

# todo 8
# pt = PlotAnalysis("E:\\Experiments\\Abosolute Threshold\\HPHTM1_27_5_ByM.nc","../Data/dazhou.shp",'HPHTM_M')
# pt.Time_Domain_Analysis_Month2()


# todo 9
# Threshold=[[7,7],[7,8],[7,9],[8,7],[8,8],[8,9],[9,7],[9,8],[9,9]]
# for i in range(len(Threshold)):
#     pt = PlotAnalysis("G:\\Experiments\\Relative Threshold\\HPHTEN3_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_ByY.nc","../Data/dazhou.shp",'HPHTEN3_Y')
#     pt.Time_Domain_Analysis_Year3()
#
# # TODO 10
# for i in range(len(Threshold)):
#     pt = PlotAnalysis("G:\\Experiments\\Relative Threshold\\HPHTEN3_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_ByM.nc","../Data/dazhou.shp",'HPHTEN3_M')
#     pt.Time_Domain_Analysis_Month3()
#
# # TODO 11
# for i in range(len(Threshold)):
#     pt = PlotAnalysis("G:\\Experiments\\Relative Threshold\\HPHTM3_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_ByY.nc","../Data/dazhou.shp",'HPHTM3_Y')
#     pt.Time_Domain_Analysis_Year2()
#
# # TODO 12
# for i in range(len(Threshold)):
#     pt = PlotAnalysis("G:\\Experiments\\Relative Threshold\\HPHTM3_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_ByM.nc","../Data/dazhou.shp",'HPHTM3_M')
#     pt.Time_Domain_Analysis_Month2()
# TODO 13
# for i in range(len(Threshold)):
#     pt = PlotAnalysis("G:\\Experiments\\Relative Threshold\\HPHTRA_P_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_ByY.nc","../Data/dazhou.shp",'t2m')
#     pt.Time_Domain_Analysis_Year3()
#
# #TODO 14
# for i in range(len(Threshold)):
#     pt = PlotAnalysis("G:\\Experiments\\Relative Threshold\\HPHTRA_T_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_ByY.nc","../Data/dazhou.shp",'__xarray_dataarray_variable__')
#     pt.Time_Domain_Analysis_Year3()
#
# # TODO 15
# for i in range(len(Threshold)):
#     pt = PlotAnalysis("G:\\Experiments\\Relative Threshold\\HPHTM3_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_ByY.nc","../Data/dazhou.shp",'HPHTM3_Y')
#     pt.Time_Domain_Analysis_Year4()

# TODO 16
# for i in range(len(Threshold)):
#     pt = PlotAnalysis("G:\\Experiments\\Relative Threshold\\HPHTM3_"+str(Threshold[i][0])+"_"+str(Threshold[i][1])+"_ByM.nc","../Data/dazhou.shp",'HPHTM3_M')
#     pt.Time_Domain_Analysis_Month4()
