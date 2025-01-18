
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




def calculate_CE_Indicator_byshape_average():
    shp_file ="D:\zhaozheng\projects\全球风险计算\SBAS_INSAR\ZZ\研究区1\多边形.shp"
    shapefile = gpd.read_file(shp_file)
    Year=np.arange(2015,2023,1).astype(str)
    Boundary=shapefile.total_bounds
    aoi_lat = [float(Boundary[1]), float(Boundary[3])]
    aoi_lon = [float(Boundary[0]), float(Boundary[2])]

    # CE_Type=['CE','CES','CET','CETS']
    # Varibale_Units=['Frequency(times)','Amplitude(days)','Duration(days)','Start(doy)','End(doy)']
    # Varibale=['Frequency','Amplitude','Duration','Start','End']
    CE_Type=['CE','CES','CET','CETS']
    Varibale_Units=['Frequency(times)']
    Varibale=['Frequency']

    # CE_Type=['CE']
    # Varibale_Units=['Ratio_PR(%)','Ratio_Tem(%)']
    # Varibale=['Ratio_PR','Ratio_Tem']
    colors=['#5FB75F','#D62728','#3A87BD','#FF7F0E']
    input_Path="E:\CE_DATA\Data_Processing\Average_Mode\\"
    Output_Path="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZZ\\Study areas\\"
    for v in range(len(Varibale)):
        data_df = pd.DataFrame({'Time':Year})
        # 设置画布大小
        fig,ax=plt.subplots()
        fig.set_size_inches(10,7)
        for t in range(len(CE_Type)):
            print([CE_Type[t]])
            filename=input_Path+CE_Type[t]+f"\\CMIP6\\{CE_Type[t]}_{Varibale[v]}_126.nc"
            dataset = xr.open_dataset(filename)
            # slice适用于至少数据大于一个分辨率的情况，适用于大区域
            data_clip = dataset[f'{CE_Type[t]}_{Varibale[v]}'].sel(
                lon=slice(aoi_lon[0], aoi_lon[1]),
                lat=slice(aoi_lat[0], aoi_lat[1])).mean(dim=['lat','lon']).sel(time=Year,quantile=0.7)
            data_clip_p=data_clip.to_pandas()
            data_df[CE_Type[t]]=data_clip_p.values
            plt.plot(data_clip_p.index, data_clip_p.values,color=colors[t],linewidth=2.5,linestyle='--',marker='*',markersize=15,label=CE_Type[t])
            # plt.plot(data_clip_p.index, data_clip_p.values,linewidth=2.5,linestyle='--',marker='*',markersize=15,label=CE_Type[t])
            # font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14}
            # for a, b in zip(data_clip_p.index, np.round(data_clip_p.values,1)):
            #     plt.text(a, b+0.5, b, ha='center', va='bottom', fontproperties=font1)
        plt.legend(loc='center', bbox_to_anchor=(0.5, 1.05), ncol=4)
        # plt.xlabel('X轴')
        plt.ylabel(f'{Varibale_Units[v]}')
            # 设置图例
        # plt.show(block = True)
        data_df.to_csv(Output_Path+F"{Varibale[v]}.csv",index=False)
        fig.savefig(Output_Path+F"{Varibale[v]}.png", dpi=800,bbox_inches='tight')



def calculate_ExtrePre_Indicator_byshape_average():
    shp_file ="D:\zhaozheng\projects\全球风险计算\SBAS_INSAR\ZZ\研究区1\多边形.shp"
    shapefile = gpd.read_file(shp_file)
    Year=np.arange(2015,2023,1).astype(str)
    # Part_Shape=shapefile[shapefile.name=="California"]# 获取部分要素
    Boundary=shapefile.total_bounds
    aoi_lat = [float(Boundary[1]), float(Boundary[3])]
    aoi_lon = [float(Boundary[0]), float(Boundary[2])]
    CE_Type=['R95P','R95D','R95C','R95DM']
    Varibale_Units=['R95P(mm)','R95D(days)','R95C(%)','R95DM(days)']
    Varibale=['Frequency','Amplitude','Duration','Start','End']
    input_Path="E:\CE_DATA\Data_Processing\Average_Mode\\"
    Output_Path="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZZ\\Study areas\\"
    data_df = pd.DataFrame({'Time':Year})
    for t in range(len(CE_Type)):
        fig,ax=plt.subplots()
        fig.set_size_inches(10,7)
        print([CE_Type[t]])
        filename=input_Path+f"R95P\\CMIP6\\PR_{CE_Type[t]}_126.nc"
        dataset = xr.open_dataset(filename)
        # slice适用于至少数据大于一个分辨率的情况，适用于大区域
        data_clip = dataset[f'{CE_Type[t]}'].sel(
            lon=slice(aoi_lon[0], aoi_lon[1]),
            lat=slice(aoi_lat[0], aoi_lat[1])).mean(dim=['lat','lon']).sel(time=Year,quantile=0.7)
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
    Year=np.arange(2015,2023,1).astype(str)
    # Part_Shape=shapefile[shapefile.name=="California"]# 获取部分要素
    Boundary=shapefile.total_bounds
    aoi_lat = [float(Boundary[1]), float(Boundary[3])]
    aoi_lon = [float(Boundary[0]), float(Boundary[2])]
    CE_Type=['T95D','T95Max_L']
    Varibale_Units=['T95D(days)','T95Max_L(days)']

    input_Path="E:\CE_DATA\Data_Processing\Average_Mode\\"
    Output_Path="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZZ\\Study areas\\"
    data_df = pd.DataFrame({'Time':Year})
    for t in range(len(CE_Type)):
        fig,ax=plt.subplots()
        fig.set_size_inches(10,7)
        print([CE_Type[t]])
        filename=input_Path+f"R95T\\{CE_Type[t]}.nc"
        dataset = xr.open_dataset(filename)
        # slice适用于至少数据大于一个分辨率的情况，适用于大区域
        data_clip = dataset[f'{CE_Type[t]}'].sel(
            longitude=slice(aoi_lon[0], aoi_lon[1]),
            latitude=slice(aoi_lat[0], aoi_lat[1])).mean(dim=['lat','lon']).sel(time=Year,quantile=0.7)
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


def calculate_Pre_TemBy_shape_average(time_range,function):
    shp_file ="D:\zhaozheng\projects\全球风险计算\SBAS_INSAR\ZZ\Landslides\Landslides.shp"
    Output_Path="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZZ\Typical Landslides\\"
    input_Path="E:\CE_DATA\Data_Processing\Average_Mode\\"
    shapefile = gpd.read_file(shp_file)
    Year=np.arange(2015,2023,1).astype(str)
    # Nums=np.arange(1,21,1)
    Nums=np.array([60,63,62,21,65,56,54,55,66,67,59,68,58,57,53,52,25,5,6,4,49,51,48,69,3,30,1,2,96,70,73,74,75,72,71,76,
          97,79,77,81,80,41,42,43,44,78,82,38,83,84,37,85,86,95,94,87,90,16,89,91,92,29,93,88])
    data_df = pd.DataFrame({'Time':time_range})
    for i in range(len(Nums)):
        print(Nums[i])
        Part_Shape=shapefile[shapefile.OBJECTID==Nums[i]]# 获取部分要素
        Boundary=Part_Shape.total_bounds
        aoi_lat = [float(Boundary[1]), float(Boundary[3])]
        aoi_lon = [float(Boundary[0]), float(Boundary[2])]
        Cal_Type=['Daily_Pre','Daily_Tem']
        Filename=['Pr','Tem']
        Variable=['pr','tasmax']
        for ct in range(len(Cal_Type)):
            print(Cal_Type[ct])
            base_filename=f"E:\CE_DATA\Data_Processing\Average_Mode\{Cal_Type[ct]}\CMIP6\{Filename[ct]}_2014.nc"
            base_file=xr.open_dataset(base_filename)
            filename=[input_Path+f"{Cal_Type[ct]}\\CMIP6\\{Filename[ct]}_{i}.nc" for  i in Year]
            datasets1=xr.open_mfdataset(filename,concat_dim="time",combine="nested",data_vars='minimal', coords='minimal', compat='override')
            datasets=xr.concat((base_file,datasets1),dim='time')
            time_Data=[]
            for tt in range(len(time_range)):
                print(tt)
                if Cal_Type[ct]=='Daily_Pre':
                    if function=='mean':
                        data_clip = datasets[f'{Variable[ct]}'].sel(
                                    lon=[aoi_lon[0], aoi_lon[1]],
                                    lat=[aoi_lat[0], aoi_lat[1]],method="nearest").sel(time=slice(time_range[tt][0],time_range[tt][1])).mean(dim=['lat','lon','time'])
                    elif function=='max':
                        data_clip = datasets[f'{Variable[ct]}'].sel(
                                    lon=[aoi_lon[0], aoi_lon[1]],
                                    lat=[aoi_lat[0], aoi_lat[1]],method="nearest").sel(time=slice(time_range[tt][0],time_range[tt][1])).max(dim=['lat','lon','time'])
                    elif function=='sum':
                        data_clip = datasets[f'{Variable[ct]}'].sel(
                                    lon=[aoi_lon[0], aoi_lon[1]],
                                    lat=[aoi_lat[0], aoi_lat[1]],method="nearest").mean(dim=['lat','lon']).sel(time=slice(time_range[tt][0],time_range[tt][1])).max(dim=['time'])
                    elif function=='effectivepre':
                        data_clip = datasets[f'{Variable[ct]}'].sel(
                                    lon=[aoi_lon[0], aoi_lon[1]],
                                    lat=[aoi_lat[0], aoi_lat[1]],method="nearest").mean(dim=['lat','lon']).sel(time=slice(time_range[tt][0],time_range[tt][1])).max(dim=['time'])
                else:
                    data_clip = datasets[f'{Variable[ct]}'].sel(
                                    lon=[aoi_lon[0], aoi_lon[1]],
                                    lat=[aoi_lat[0], aoi_lat[1]],method="nearest").sel(time=slice(time_range[tt][0],time_range[tt][1])).max(dim=['lat','lon','time'])
                time_Data.append(data_clip.values)
            data_df[f'{str(Nums[i])}_{Cal_Type[ct]}']=time_Data
    data_df.to_csv(f'{Output_Path}Pre_Tem_{function}.csv',index=False)

def calculate_Pre_TemBy_shape_average3(time_range,function):
    shp_file ="D:\zhaozheng\projects\全球风险计算\SBAS_INSAR\ZZ\Landslides\Landslides.shp"
    Output_Path="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZZ\Typical Landslides\\"
    input_Path="E:\CE_DATA\Data_Processing\Average_Mode\\"
    shapefile = gpd.read_file(shp_file)
    Year=np.arange(2015,2023,1).astype(str)
    # Nums=np.arange(1,21,1)
    # Nums=np.array([60,63,62,21,65,56,54,55,66,67,59,68,58,57,53,52,25,5,6,4,49,51,48,69,3,30,1,2,96,70,73,74,75,72,71,76,
    #       97,79,77,81,80,41,42,43,44,78,82,38,83,84,37,85,86,95,94,87,90,16,89,91,92,29,93,88])

    Nums=np.array([58,57,53,52,5,6,4,3,30,1,2,41,42,84,90,16,89,92,29])

    # Nums=np.array([60])
    data_df = pd.DataFrame({'Time':time_range})
    Cal_Type=['Daily_Pre','Daily_Tem']
    Filename=['Pr','Tem']
    Variable=['pr','tasmax']
    for ct in range(len(Cal_Type)):
        # print(Cal_Type[ct])
        base_filename=f"E:\CE_DATA\Data_Processing\Average_Mode\\{Cal_Type[ct]}\CMIP6\\{Filename[ct]}_2014.nc"
        base_file=xr.open_dataset(base_filename)
        filename=[input_Path+f"{Cal_Type[ct]}\\CMIP6\\{Filename[ct]}_{i}.nc" for  i in Year]
        datasets1=xr.open_mfdataset(filename,concat_dim="time",combine="nested",data_vars='minimal', coords='minimal', compat='override')
        datasets=xr.concat((base_file,datasets1),dim='time')
        for i in range(len(Nums)):
            print(Nums[i])
            Part_Shape=shapefile[shapefile.OBJECTID==Nums[i]]# 获取部分要素
            Boundary=Part_Shape.total_bounds
            aoi_lat = [float(Boundary[1]), float(Boundary[3])]
            aoi_lon = [float(Boundary[0]), float(Boundary[2])]
            time_Data=[]
            for tt in range(len(time_range)):
                if Cal_Type[ct]=='Daily_Pre':
                    if function=='mean':
                        data_clip = datasets[f'{Variable[ct]}'].sel(
                                    lon=[aoi_lon[0], aoi_lon[1]],
                                    lat=[aoi_lat[0], aoi_lat[1]],method="nearest").sel(time=slice(time_range[tt][0],time_range[tt][1])).mean(dim=['lat','lon','time'])
                        time_Data.append(data_clip.values)
                    elif function=='max':
                        data_clip = datasets[f'{Variable[ct]}'].sel(
                                    lon=[aoi_lon[0], aoi_lon[1]],
                                    lat=[aoi_lat[0], aoi_lat[1]],method="nearest").sel(time=slice(time_range[tt][0],time_range[tt][1])).max(dim=['lat','lon','time'])
                        time_Data.append(data_clip.values)
                    elif function=='sum':
                        data_clip = datasets[f'{Variable[ct]}'].sel(
                                    lon=[aoi_lon[0], aoi_lon[1]],
                                    lat=[aoi_lat[0], aoi_lat[1]],method="nearest").mean(dim=['lat','lon']).sel(time=slice(time_range[tt][0],time_range[tt][1])).sum(dim=['time'])
                        time_Data.append(data_clip.values)
                    elif function=='effectivepre':
                        data_clip_nu = datasets[f'{Variable[ct]}'].sel(
                                    lon=[aoi_lon[0], aoi_lon[1]],
                                    lat=[aoi_lat[0], aoi_lat[1]],method="nearest").mean(dim=['lat','lon']).sel(time=slice(time_range[tt][0],time_range[tt][1])).to_numpy()
                        data_clip=0
                        data_clip_nu_re = np.flipud(data_clip_nu)
                        for iii in range(data_clip_nu_re.shape[0]):
                            data_clip=data_clip+np.power(0.9,iii)*data_clip_nu_re[iii]
                        time_Data.append(data_clip)
                else:
                    data_clip = datasets[f'{Variable[ct]}'].sel(
                                    lon=[aoi_lon[0], aoi_lon[1]],
                                    lat=[aoi_lat[0], aoi_lat[1]],method="nearest").sel(time=slice(time_range[tt][0],time_range[tt][1])).mean(dim=['lat','lon','time'])
                    time_Data.append(data_clip.values)
            data_df[f'{str(Nums[i])}_{Cal_Type[ct]}']=time_Data
    data_df.to_csv(f'{Output_Path}Effect_Pre_Tem_{function}.csv',index=False)


deformation=pd.read_csv("D:\zhaozheng\projects\全球风险计算\SBAS_INSAR\ZZ\Deformation\Deformation.csv")
# time_pairs = [(deformation['Times'][i], deformation['Times'][i+1]) for i in range(len(deformation['Times'])-1)]
# 按照键值对算，按照15日有效算
# 有效降雨量计算
deformation['Times']=pd.to_datetime(deformation['Times'],format='%Y-%m-%d')
time_pairs_lag= [(deformation['Times'][i] - pd.Timedelta(days=15), deformation['Times'][i]) for i in range(len(deformation['Times'])-1)]
calculate_Pre_TemBy_shape_average3(time_pairs_lag,'mean')
calculate_Pre_TemBy_shape_average3(time_pairs_lag,'max')
calculate_Pre_TemBy_shape_average3(time_pairs_lag,'sum')
calculate_Pre_TemBy_shape_average3(time_pairs_lag,'effectivepre')






def calculate_ThresholdBy_shape_average():
    shp_file ="D:\zhaozheng\projects\全球风险计算\SBAS_INSAR\ZGH\\7_27\\Landslides\\ALL_Landslides.shp"
    Output_Path="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZGH\Typical_Landslides\\"
    input_Path="E:\CE_DATA\Data_Processing\Average_Mode\Threshold\CMIP6\\"
    Variable=['Pr','Tem']
    shapefile = gpd.read_file(shp_file)
    Nums=np.arange(1,21,1)
    data_df = pd.DataFrame()
    for i in range(len(Nums)):
        Part_Shape=shapefile[shapefile.Id==Nums[i]]# 获取部分要素
        Boundary=Part_Shape.total_bounds
        aoi_lat = [float(Boundary[1]), float(Boundary[3])]
        aoi_lon = [float(Boundary[0]), float(Boundary[2])]
        for ct in range(len(Variable)):
            print(Variable[ct])
            filename=input_Path+f"{Variable[ct]}_Threshold.nc"
            datasets=xr.open_dataset(filename)
            data_clip = datasets['Threshold'].sel(
                        lon=[aoi_lon[0], aoi_lon[1]],
                        lat=[aoi_lat[0], aoi_lat[1]],method="nearest").mean(dim=['lat','lon'])

            data_df[f'F{str(Nums[i])}_{Variable[ct]}']=data_clip.values
    data_df.to_csv(Output_Path+F"Threshold_Pre_Tem.csv",index=False)
def calculate_Pre_TemBy_shape_average2(time_range):
    shp_file ="D:\zhaozheng\projects\全球风险计算\SBAS_INSAR\ZGH\\7_27\\Landslides\\ALL_Landslides.shp"
    Output_Path="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZGH\Typical_Landslides\\"
    input_Path="E:\CE_DATA\Data_Processing\Average_Mode\\"
    shapefile = gpd.read_file(shp_file)
    Year=np.arange(2015,2024,1).astype(str)
    Nums=np.arange(1,21,1)
    data_df = pd.DataFrame({'Time':time_range})
    for i in range(len(Nums)):
        print(Nums[i])
        Part_Shape=shapefile[shapefile.Id==Nums[i]]# 获取部分要素
        Boundary=Part_Shape.total_bounds
        aoi_lat = [float(Boundary[1]), float(Boundary[3])]
        aoi_lon = [float(Boundary[0]), float(Boundary[2])]
        Cal_Type=['Daily_Pre','Daily_Tem']
        Filename=['Pr','Tem']
        Variable=['pr','tasmax']
        for ct in range(len(Cal_Type)):
            print(Cal_Type[ct])
            filename=[input_Path+f"{Cal_Type[ct]}\\CMIP6\\{Filename[ct]}_{i}.nc" for  i in Year]
            datasets=xr.open_mfdataset(filename,concat_dim="time",combine="nested",data_vars='minimal', coords='minimal', compat='override')
            time_Data=[]
            for tt in range(len(time_range)):
                data_clip = datasets[f'{Variable[ct]}'].sel(
                            lon=[aoi_lon[0], aoi_lon[1]],
                            lat=[aoi_lat[0], aoi_lat[1]],method="nearest").sel(time=slice(time_range[tt][0],time_range[tt][1])).mean(dim=['lat','lon','time'])
                time_Data.append(data_clip.values)
            data_df[f'F{str(Nums[i])}_{Cal_Type[ct]}']=time_Data
    data_df.to_csv(Output_Path+F"Pre_Tem_mean.csv",index=False)



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





