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
    csv_file="D:\zhaozheng\projects\全球风险计算\SBAS_INSAR\ZGH\zgh_2024\Rock_avalanche.csv"
    Deformation_data=pd.read_csv(csv_file)
    Deformation_data['Date']=pd.to_datetime(Deformation_data['Date'],format='%Y/%m/%d')
    delate_days_array=[0]
    Deformation_Vectory_array=np.zeros([79,1])
    for t in range(Deformation_data['Date'].shape[0]-1):
        delate_days=pd.Timedelta(Deformation_data['Date'][t+1] - Deformation_data['Date'][t]).days
        delate_days_array.append(delate_days)
        Deformation_Vectory=(Deformation_data[Deformation_data.columns[1:]].values[t+1,:]-Deformation_data[Deformation_data.columns[1:]].values[t,:])/delate_days
        Deformation_Vectory_array=np.concatenate((Deformation_Vectory_array,Deformation_Vectory.reshape(79,1)),axis=1)
    Deformation_data['Delate_Day']=delate_days_array
    Columns_de=[f'{i}v'for i in Deformation_data.columns[1:-1]]
    for cd in range(len(Columns_de)):
        Deformation_data[Columns_de[cd]]=Deformation_Vectory_array[cd]
    Deformation_data.to_csv('D:\zhaozheng\projects\全球风险计算\SBAS_INSAR\ZGH\zgh_2024\Rock_avalanche_All.csv',index=False)
    return Deformation_data




def calculate_Pre_TemBy_shape_average(lag,land_num,time_range_start,time_range_End):
    shp_file ="D:\zhaozheng\projects\全球风险计算\SBAS_INSAR\ZGH\zgh_2024\RA\RA.shp"
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
        np.savetxt(Output_Path+F"RA_{Cal_Type[ct]}_max{lag}.csv", array_type, delimiter=',')

def calculate_Pre_TemBy_shape_average2():
    shp_file ="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZGH\Study_area\BLA_P.shp"
    Output_Path="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZGH\Study_area\\results\\"
    input_Path="E:\CE_DATA\Data_Processing\Average_Mode\Daily_Pre\ERA5\\"
    shapefile = gpd.read_file(shp_file)
    Variable=['tp','mx2t']
    # filename=["E:\CE_DATA\Data_Processing\Average_Mode\Daily_Pre\ERA5\\total_precipitation_2023.nc",
    # "E:\CE_DATA\Data_Processing\Average_Mode\Daily_Pre\ERA5\\total_precipitation_2021.nc",
    # "E:\CE_DATA\Data_Processing\Average_Mode\Daily_Pre\ERA5\\total_precipitation_2022.nc"]
    filename=["E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2022.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2023.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2021.nc"]
    datasets=xr.open_mfdataset(filename,concat_dim="time",combine="nested",data_vars='minimal', coords='minimal', compat='override')
    aoi_lat_array = []
    aoi_lon_array=[]
    Part_Shape=shapefile[shapefile.OBJECTID==1]# 获取部分要素
    Boundary=Part_Shape.total_bounds
    aoi_lat_array.append(float(Boundary[1]))
    # aoi_lat_array.append(float(Boundary[3]))
    aoi_lon_array.append(float(Boundary[0]))
    # aoi_lon_array.append(float(Boundary[2]))
    data_clip = datasets['mx2t'].sel(
                    longitude=aoi_lon_array,
                    latitude=aoi_lat_array,method="nearest").max(dim=['longitude','latitude'])
    # data_clip=data_clip*1000
    data_clip=data_clip - 273.15
    np.savetxt(Output_Path+F"Tem_2021_2023.csv", data_clip.to_numpy(), delimiter=',')
def calculate_Pre_TemBy_shape_average3():
    Output_Path="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZGH\Study_area\\results\\"
    filename=["E:\CE_DATA\Data_Processing\Average_Mode\Daily_Pre\ERA5\\total_precipitation_2023.nc",
    "E:\CE_DATA\Data_Processing\Average_Mode\Daily_Pre\ERA5\\total_precipitation_2021.nc",
    "E:\CE_DATA\Data_Processing\Average_Mode\Daily_Pre\ERA5\\total_precipitation_2022.nc"]
    datasets=xr.open_mfdataset(filename,concat_dim="time",combine="nested",data_vars='minimal', coords='minimal', compat='override')
    data_clip = datasets['tp'].groupby('time.year').sum(dim='time').mean(dim='year')
    data_clip=data_clip*1000
    # data_clip=data_clip - 273.15
    data_clip.to_netcdf(Output_Path+f"pre.nc")

def calculate_Pre_TemBy_shape_average3():
    shp_file ="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZGH\Study_area\BLA_P.shp"
    Output_Path="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZGH\Study_area\\results\\"
    shapefile = gpd.read_file(shp_file)
    filename=["E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1950.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1951.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1952.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1953.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1954.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1955.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1956.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1957.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1958.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1959.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1960.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1961.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1962.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1963.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1964.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1965.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1966.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1967.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1968.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1969.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1970.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1971.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1972.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1973.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1974.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1975.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1976.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1977.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1978.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1979.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1980.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1981.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1982.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1983.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1984.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1985.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1986.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1987.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1988.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1989.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1990.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1991.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1992.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1993.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1994.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1995.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1996.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1997.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1998.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1999.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2000.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2001.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2002.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2003.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2004.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2005.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2006.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2007.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2008.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2009.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2010.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2011.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2012.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2013.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2014.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2015.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2016.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2017.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2018.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2019.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2020.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2021.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2022.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2023.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1941.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1942.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1943.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1944.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1945.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1946.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1947.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1948.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_1949.nc"]
    datasets=xr.open_mfdataset(filename,concat_dim="time",combine="nested",data_vars='minimal', coords='minimal', compat='override')
    aoi_lat_array = []
    aoi_lon_array=[]
    Part_Shape=shapefile[shapefile.OBJECTID==1]# 获取部分要素
    Boundary=Part_Shape.total_bounds
    aoi_lat_array.append(float(Boundary[1]))
    # aoi_lat_array.append(float(Boundary[3]))
    aoi_lon_array.append(float(Boundary[0]))
    # aoi_lon_array.append(float(Boundary[2]))

    data_clip = datasets['mx2t'].groupby('time.year').mean(dim='time').sel(
                    longitude=aoi_lon_array,
                    latitude=aoi_lat_array,method="nearest").mean(dim=['longitude','latitude'])
    # data_clip=data_clip*1000
    data_clip=data_clip - 273.15
    np.savetxt(Output_Path+F"Tem_Year2.csv", data_clip.to_numpy(), delimiter=',')




def cal_daily_pre_tem():
    shp_file ="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZGH\Typical_Landslides2\\10_23\RA.shp"
    Output_Path="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZGH\Typical_Landslides2\\"
    shapefile = gpd.read_file(shp_file)
    filename=["E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2021.nc",
        "E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2022.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2023.nc"]
    datasets=xr.open_mfdataset(filename,concat_dim="time",combine="nested",data_vars='minimal', coords='minimal', compat='override')
    Nums=np.arange(0,87,1)
    array_type= np.zeros((datasets.time.shape[0], 88), dtype=float)
    array_type[:,0]=datasets.time.to_numpy()
    for i in range(len(Nums)):
        print(i)
        aoi_lat_array = []
        aoi_lon_array=[]
        Part_Shape=shapefile[shapefile.ID==Nums[i]]# 获取部分要素
        Boundary=Part_Shape.total_bounds
        aoi_lat_array.append(float(Boundary[1]))
        aoi_lon_array.append(float(Boundary[0]))
        data_clip = datasets['mx2t'].sel(
                        longitude=aoi_lon_array,
                        latitude=aoi_lat_array,method="nearest").max(dim=['longitude','latitude'])
        #data_clip=data_clip*1000
        data_clip=data_clip - 273.15
        array_type[:,i+1]=data_clip.to_numpy()
    np.savetxt(Output_Path+F"Tem_RA.csv", array_type, delimiter=',')

def cal_daily_pre_tem2():
    shp_file ="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZGH\Typical_Landslides2\\10_23\RF.shp"
    Output_Path="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZGH\Typical_Landslides2\\"
    shapefile = gpd.read_file(shp_file)
    filename=["E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2021.nc",
        "E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2022.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2023.nc"]
    datasets=xr.open_mfdataset(filename,concat_dim="time",combine="nested",data_vars='minimal', coords='minimal', compat='override')
    Nums=np.arange(0,66,1)
    array_type= np.zeros((datasets.time.shape[0], 67), dtype=float)
    array_type[:,0]=datasets.time.to_numpy()
    for i in range(len(Nums)):
        print(i)
        aoi_lat_array = []
        aoi_lon_array=[]
        Part_Shape=shapefile[shapefile.ID==Nums[i]]# 获取部分要素
        Boundary=Part_Shape.total_bounds
        aoi_lat_array.append(float(Boundary[1]))
        aoi_lon_array.append(float(Boundary[0]))
        data_clip = datasets['mx2t'].sel(
                        longitude=aoi_lon_array,
                        latitude=aoi_lat_array,method="nearest").max(dim=['longitude','latitude'])
        #data_clip=data_clip*1000
        data_clip=data_clip - 273.15
        array_type[:,i+1]=data_clip.to_numpy()
    np.savetxt(Output_Path+F"Tem_RF.csv", array_type, delimiter=',')

def cal_daily_pre_tem3():
    shp_file ="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZGH\Typical_Landslides2\\10_23\RG.shp"
    Output_Path="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZGH\Typical_Landslides2\\"
    shapefile = gpd.read_file(shp_file)
    filename=["E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2021.nc",
        "E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2022.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2023.nc"]
    datasets=xr.open_mfdataset(filename,concat_dim="time",combine="nested",data_vars='minimal', coords='minimal', compat='override')
    Nums=np.arange(0,253,1)
    array_type= np.zeros((datasets.time.shape[0], 254), dtype=float)
    array_type[:,0]=datasets.time.to_numpy()
    for i in range(len(Nums)):
        print(i)
        aoi_lat_array = []
        aoi_lon_array=[]
        Part_Shape=shapefile[shapefile.ID==Nums[i]]# 获取部分要素
        Boundary=Part_Shape.total_bounds
        aoi_lat_array.append(float(Boundary[1]))
        aoi_lon_array.append(float(Boundary[0]))
        data_clip = datasets['mx2t'].sel(
                        longitude=aoi_lon_array,
                        latitude=aoi_lat_array,method="nearest").max(dim=['longitude','latitude'])
        #data_clip=data_clip*1000
        data_clip=data_clip - 273.15
        array_type[:,i+1]=data_clip.to_numpy()
    np.savetxt(Output_Path+F"Tem_RG.csv", array_type, delimiter=',')

def cal_daily_pre_tem4():
    shp_file ="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZGH\Typical_Landslides2\\10_23\RSD.shp"
    Output_Path="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZGH\Typical_Landslides2\\"
    shapefile = gpd.read_file(shp_file)
    filename=["E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2021.nc",
        "E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2022.nc",
"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\ERA5\Maximum 2m temperature since previous post-processing_2023.nc"]
    datasets=xr.open_mfdataset(filename,concat_dim="time",combine="nested",data_vars='minimal', coords='minimal', compat='override')
    Nums=np.arange(0,93,1)
    array_type= np.zeros((datasets.time.shape[0], 94), dtype=float)
    array_type[:,0]=datasets.time.to_numpy()
    for i in range(len(Nums)):
        print(i)
        aoi_lat_array = []
        aoi_lon_array=[]
        Part_Shape=shapefile[shapefile.ID==Nums[i]]# 获取部分要素
        Boundary=Part_Shape.total_bounds
        aoi_lat_array.append(float(Boundary[1]))
        aoi_lon_array.append(float(Boundary[0]))
        data_clip = datasets['mx2t'].sel(
                        longitude=aoi_lon_array,
                        latitude=aoi_lat_array,method="nearest").max(dim=['longitude','latitude'])
        #data_clip=data_clip*1000
        data_clip=data_clip - 273.15
        array_type[:,i+1]=data_clip.to_numpy()
    np.savetxt(Output_Path+F"Tem_RSD.csv", array_type, delimiter=',')

cal_daily_pre_tem()
cal_daily_pre_tem2()
cal_daily_pre_tem3()
cal_daily_pre_tem4()

# Deformation_data=Read_Csv_Defomation_ve()
# time_day_lat=np.arange(1,15,1)
# for i in range(time_day_lat.shape[0]):
#     time_range_start=Deformation_data['Date'] - pd.Timedelta(days=time_day_lat[i])
#     time_range_End=Deformation_data['Date']
#     calculate_Pre_TemBy_shape_average(i,int((Deformation_data.shape[1]-2)/2),time_range_start,time_range_End)

