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
import scipy.stats


def Read_Csv_Defomation_ve():
    csv_file="D:\zhaozheng\projects\全球风险计算\SBAS_INSAR\ZGH\zgh_2024\Rockslide.csv"
    Deformation_data=pd.read_csv(csv_file)
    Deformation_data['Date']=pd.to_datetime(Deformation_data['Date'],format='%Y/%m/%d')
    delate_days_array=[0]
    Deformation_Vectory_array=np.zeros([67,1])
    for t in range(Deformation_data['Date'].shape[0]-1):
        delate_days=pd.Timedelta(Deformation_data['Date'][t+1] - Deformation_data['Date'][t]).days
        delate_days_array.append(delate_days)
        Deformation_Vectory=(Deformation_data[Deformation_data.columns[1:]].values[t+1,:]-Deformation_data[Deformation_data.columns[1:]].values[t,:])/delate_days
        Deformation_Vectory_array=np.concatenate((Deformation_Vectory_array,Deformation_Vectory.reshape(67,1)),axis=1)
    Deformation_data['Delate_Day']=delate_days_array
    Columns_de=[f'{i}v'for i in Deformation_data.columns[1:-1]]
    for cd in range(len(Columns_de)):
        Deformation_data[Columns_de[cd]]=Deformation_Vectory_array[cd]
    Deformation_data.to_csv('D:\zhaozheng\projects\全球风险计算\SBAS_INSAR\ZGH\zgh_2024\Rockslide_All.csv',index=False)
    return Deformation_data

def calculate_Pre_TemBy_shape_average(lag,land_num,time_range_start,time_range_End):
    shp_file ="D:\zhaozheng\projects\全球风险计算\SBAS_INSAR\ZGH\zgh_2024\RS\RS.shp"
    Output_Path="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZGH\Typical_Landslides\\"
    input_Path="E:\CE_DATA\Data_Processing\Average_Mode\\"
    shapefile = gpd.read_file(shp_file)
    Year=np.arange(2021,2024,1).astype(str)
    Nums=np.arange(1,land_num+1,1)
    data_df = pd.DataFrame({'Time':time_range_End})
    Cal_Type=['Daily_Pre','Daily_Tem']
    Filename=['total_precipitation','Maximum 2m temperature since previous post-processing']
    Variable=['tp','mx2t']
    for ct in range(len(Cal_Type)):
        print(Cal_Type[ct])
        array_type= np.zeros((time_range_start.shape[0], land_num), dtype=float)
        filename=[input_Path+f"{Cal_Type[ct]}\\ERA5\\{Filename[ct]}_{i}.nc" for  i in Year]
        datasets=xr.open_mfdataset(filename,concat_dim="time",combine="nested",data_vars='minimal', coords='minimal', compat='override')
        aoi_lat_array = []
        aoi_lon_array=[]
        for i in range(len(Nums)):
            Part_Shape=shapefile[shapefile.ID==Nums[i]-1]# 获取部分要素
            Boundary=Part_Shape.total_bounds
            aoi_lat_array.append(float(Boundary[1]))
            # aoi_lat_array.append(float(Boundary[3]))
            aoi_lon_array.append(float(Boundary[0]))
            # aoi_lon_array.append(float(Boundary[2]))
            time_Data=[]
        for tt in range(time_range_start.shape[0]):
            print(tt)
            data_clip = datasets[f'{Variable[ct]}'].sel(
                        longitude=aoi_lon_array,
                        latitude=aoi_lat_array,method="nearest").sel(time=slice(time_range_start[tt].strftime('%Y-%m-%d'),time_range_End[tt].strftime('%Y-%m-%d'))).max(dim=['time','longitude'])
            if ct==0:
                data_clip=data_clip*1000
            else:
                data_clip=data_clip - 273.15
            array_type[tt,:]=data_clip.to_numpy()
        np.savetxt(Output_Path+F"RS_{Cal_Type[ct]}_max{lag}.csv", array_type, delimiter=',')

Deformation_data=Read_Csv_Defomation_ve()
time_day_lat=np.arange(1,15,1)
# for i in range(time_day_lat.shape[0]):
#     time_range_start=Deformation_data['Date'] - pd.Timedelta(days=time_day_lat[i])
#     time_range_End=Deformation_data['Date']
#     calculate_Pre_TemBy_shape_average(i,int((Deformation_data.shape[1]-2)/2),time_range_start,time_range_End)
time_range_start=Deformation_data['Date'] - pd.Timedelta(days=15)
time_range_End=Deformation_data['Date']
calculate_Pre_TemBy_shape_average(14,int((Deformation_data.shape[1]-2)/2),time_range_start,time_range_End)
# scipy.stats.pearsonr(x,y)
