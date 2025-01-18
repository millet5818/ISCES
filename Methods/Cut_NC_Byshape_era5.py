
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
import geopandas as gpd
# import earthpy as et
import xarray as xr
# .nc文件的空间切片包
import regionmask
from osgeo import gdal
from netCDF4 import Dataset
from pyproj import Proj, transform
from dask.diagnostics import ProgressBar
import matplotlib.pyplot  as plt
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = ["Times New Roman"] # TODO  Windows_Functions
mpl.rcParams['font.size'] = 18# 设置全局字体大小

shp_file ="D:\zhaozheng\projects\全球风险计算\SBAS_INSAR\ZZ\研究区1\多边形.shp"
shapefile = gpd.read_file(shp_file)
Year=np.arange(2014,2023,1).astype(str)
# Part_Shape=shapefile[shapefile.name=="California"]# 获取部分要素
Boundary=shapefile.total_bounds
aoi_lat = [float(Boundary[1]), float(Boundary[3])]
aoi_lon = [float(Boundary[0]), float(Boundary[2])]

CE_Type=['CE','CES','CET','CETS']
Varibale_Units=['Frequency(times)','Amplitude(days)','Duration(days)','Start(doy)','End(doy)']
Varibale=['Frequency','Amplitude','Duration','Start','End']

# CE_Type=['CE']
# Varibale_Units=['Ratio_PR(%)','Ratio_Tem(%)']
# Varibale=['Ratio_PR','Ratio_Tem']



def calculate_CE_Indicator_byshape_average():
    input_Path="E:\CE_DATA\Data_Processing\Average_Mode\\"
    Output_Path="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZZ\Study areas\\"
    for v in range(len(Varibale)):
        data_df = pd.DataFrame({'Time':Year})
        # 设置画布大小
        fig,ax=plt.subplots()
        fig.set_size_inches(10,7)
        for t in range(len(CE_Type)):
            print([CE_Type[t]])
            filename=input_Path+CE_Type[t]+f"\\ERA5\\CE_{Varibale[v]}_ERA5.nc"
            dataset = xr.open_dataset(filename)
            # slice适用于至少数据大于一个分辨率的情况，适用于大区域
            data_clip = dataset[f'CE_{Varibale[v]}'].sel(
                longitude=slice(aoi_lon[0], aoi_lon[1]),
                latitude=slice(aoi_lat[0], aoi_lat[1])).mean(dim=['latitude','longitude']).sel(time=Year,quantile=0.7)
            data_clip_p=data_clip.to_pandas()
            data_df[CE_Type[t]]=data_clip_p.values
            plt.plot(data_clip_p.index, data_clip_p.values,linewidth=2.5,linestyle='--',marker='*',markersize=15,label=CE_Type[t])
        plt.legend(loc='center', bbox_to_anchor=(0.5, 1.05), ncol=4)
        # plt.xlabel('X轴')
        plt.ylabel(f'{Varibale_Units[v]}')
            # 设置图例
        # plt.show(block = True)
        data_df.to_csv(Output_Path+F"{Varibale[v]}.csv",index=False)
        fig.savefig(Output_Path+F"{Varibale[v]}.png", dpi=800,bbox_inches='tight')
        print(1)
def calculate_ExtrePre_Indicator_byshape_average():
    shp_file ="D:\zhaozheng\projects\全球风险计算\SBAS_INSAR\ZZ\研究区1\多边形.shp"
    shapefile = gpd.read_file(shp_file)
    Year=np.arange(2014,2023,1).astype(str)
    # Part_Shape=shapefile[shapefile.name=="California"]# 获取部分要素
    Boundary=shapefile.total_bounds
    aoi_lat = [float(Boundary[1]), float(Boundary[3])]
    aoi_lon = [float(Boundary[0]), float(Boundary[2])]
    CE_Type=['R95D','R95C','R95DM','R95P']
    filename_a=['R95p','R95C','R95DM','R95P']
    Varibale_Units=['R95D(days)','R95C(%)','R95DM(days)','R95P(mm)']
    input_Path="E:\CE_DATA\Data_Processing\Average_Mode\\"
    Output_Path="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZZ\\Study areas\ERA5\\"

    data_df = pd.DataFrame({'Time':Year})
    for t in range(len(CE_Type)):
        fig,ax=plt.subplots()
        fig.set_size_inches(10,7)
        print([CE_Type[t]])
        filename=input_Path+f"R95P\\ERA5\\{CE_Type[t]}.nc"
        dataset = xr.open_dataset(filename)
        # slice适用于至少数据大于一个分辨率的情况，适用于大区域
        data_clip = dataset[f'{filename_a[t]}'].sel(
            longitude=slice(aoi_lon[0], aoi_lon[1]),
            latitude=slice(aoi_lat[0], aoi_lat[1])).mean(dim=['latitude','longitude']).sel(time=Year,quantile=0.7)
        data_clip_p=data_clip.to_pandas()
        data_df[CE_Type[t]]=data_clip_p.values
        plt.plot(data_clip_p.index, data_clip_p.values,linewidth=2.5,linestyle='--',marker='*',markersize=15,label=CE_Type[t])
        plt.legend(loc='center', bbox_to_anchor=(0.5, 1.05), ncol=4)
    # plt.xlabel('X轴')
        plt.ylabel(f'{Varibale_Units[t]}')
        fig.savefig(Output_Path+F"{CE_Type[t]}.png", dpi=800,bbox_inches='tight')
        # 设置图例
    # plt.show(block = True)
    data_df.to_csv(Output_Path+F"Extreme_Pre.csv",index=False)
def calculate_ExtreTem_Indicator_byshape_average():
    shp_file ="D:\zhaozheng\projects\全球风险计算\SBAS_INSAR\ZZ\研究区1\多边形.shp"
    shapefile = gpd.read_file(shp_file)
    Year=np.arange(2014,2023,1).astype(str)
    # Part_Shape=shapefile[shapefile.name=="California"]# 获取部分要素
    Boundary=shapefile.total_bounds
    aoi_lat = [float(Boundary[1]), float(Boundary[3])]
    aoi_lon = [float(Boundary[0]), float(Boundary[2])]
    CE_Type=['T95D','T95Max_L','T95End_T','T95Start_T','T95Events']
    Varibale_Units=['T95D(days)','T95Max_L(days)','T95End_T(doy)','T95Start_T(doy)','T95Events(times)']
    filename_a=['T90D','T95Max_L','dayofyear','dayofyear','T95Events']
    input_Path="E:\CE_DATA\Data_Processing\Average_Mode\\"
    Output_Path="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZZ\\Study areas\ERA5\\"
    data_df = pd.DataFrame({'Time':Year})
    for t in range(len(CE_Type)):
        fig,ax=plt.subplots()
        fig.set_size_inches(10,7)
        print([CE_Type[t]])
        filename=input_Path+f"R95T\\{CE_Type[t]}.nc"
        dataset = xr.open_dataset(filename)
        # slice适用于至少数据大于一个分辨率的情况，适用于大区域
        data_clip = dataset[f'{filename_a[t]}'].sel(
            longitude=slice(aoi_lon[0], aoi_lon[1]),
            latitude=slice(aoi_lat[0], aoi_lat[1])).mean(dim=['latitude','longitude']).sel(time=Year,quantile=0.7)
        data_clip_p=data_clip.to_pandas()
        data_df[CE_Type[t]]=data_clip_p.values
        plt.plot(data_clip_p.index, data_clip_p.values,linewidth=2.5,linestyle='--',marker='*',markersize=15,label=CE_Type[t])
        plt.legend(loc='center', bbox_to_anchor=(0.5, 1.05), ncol=4)
    # plt.xlabel('X轴')
        plt.ylabel(f'{Varibale_Units[t]}')
        fig.savefig(Output_Path+F"{CE_Type[t]}.png", dpi=800,bbox_inches='tight')
        # 设置图例
    # plt.show(block = True)
    data_df.to_csv(Output_Path+F"Extreme_Tem.csv",index=False)

calculate_ExtreTem_Indicator_byshape_average()













# 因为数据少于一个分辨率，所以只能插值,适用于很小的滑坡
# two_months_cali = dataset["CE_Amplitude"].sel(
#     lon=[aoi_lon[0], aoi_lon[1]],
#     lat=[aoi_lat[0], aoi_lat[1]],method="nearest").mean(dim=['lat','lon']).sel(time=Year)
# print(1)

# # 创造mask,裁剪nc 并保存nc
# cali_mask=regionmask.mask_3D_geopandas(shapefile,
#                                       dataset["CE_Amplitude"].lon,
#                                       dataset["CE_Amplitude"].lat)
#
# dataset_mask=dataset["CE_Amplitude"].where(cali_mask)
# delayed_obj =dataset_mask.to_netcdf(output_file, format='NETCDF4', engine='netcdf4',mode='w',compute=False)
# with ProgressBar():
#     results222 = delayed_obj.compute()





