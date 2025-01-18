"""
绘制Historical空间分布图

"""

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
from cartopy.io.shapereader import Reader
from matplotlib import colors
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from matplotlib.ticker  import MaxNLocator
mpl.rcParams["font.sans-serif"] = ["Times New Roman"] # TODO  Windows_Functions
mpl.rcParams['font.size'] = 16# 设置全局字体大小

FilePath=['CE_ACCESS-CM2','CE_BCC-CSM2-MR','CE_CanESM5','CE_CMCC-ESM2','CE_GFDL-ESM4','CE_IPSL-CM6A-LR','CE_MIROC6','CE_NESM3']
Mode=['Historical','126','245','585']

def _forward(x):
    return np.log10(np.sqrt(x**2+1)+x)
def _inverse(x):
    return np.log10(np.sqrt(x**2+1)+x)

def Spatial_By_Average(shapefile,OutputFile_P,data,j,color):
    print(1)
    mpl.rcParams['font.size'] = 24
    if (j==4 or j ==7):
        CE_Frequency_Map= data
    else:
        CE_Frequency_Map = data.mean(dim='time')
    data_Variables = list(CE_Frequency_Map.data_vars.keys())
    if j==5 or j ==6:
        data_var = sum([CE_Frequency_Map[i] for i in data_Variables]) / len(list(CE_Frequency_Map.data_vars.keys()))*100
    else:
        data_var = sum([CE_Frequency_Map[i] for i in data_Variables]) / len(list(CE_Frequency_Map.data_vars.keys()))
    vmin = np.nanmin(data_var)
    vmax = np.nanmax(data_var)
    if j ==5 or j ==6:
        common_levels = np.linspace(vmin, vmax, 9)
        norm = colors.BoundaryNorm(common_levels, color.N)
    elif j==7:
        common_levels = np.linspace(vmin, vmax, 9)
        norm = colors.BoundaryNorm(common_levels, color.N)
    elif j==4:
        common_levels = np.array([vmin, 1,2,3,5,10,20,40,vmax])
        norm = colors.SymLogNorm(linthresh=0.5, linscale=0.5, vmin=vmin, vmax=vmax, base=20)
    else:
        common_levels = np.linspace(vmin, vmax, 10).astype(int)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    # norm = colors.FuncNorm((_forward, _inverse), vmin=vmin, vmax=vmax)
    # norm = colors.SymLogNorm(linthresh=0.5, linscale=0.5, vmin=vmin, vmax=vmax, base=20)
    longitude = data_var.lon
    latitude = data_var.lat
    gdf = gpd.read_file(shapefile)
    Threshold=np.array([0.7,0.8,0.9])
    for i in range(len(Threshold)):
        data_single = data_var.sel(quantile=Threshold[i])
        maskregion1 = data_single.salem.roi(shape=gdf)
        data1 = maskregion1.values
        fig1 = plt.figure(figsize=(16, 9))
        proj = ccrs.PlateCarree()
        ax1 = fig1.add_subplot(111, projection=ccrs.PlateCarree())
        china_map = cfeature.ShapelyFeature(shpreader.Reader(shapefile).geometries(), proj, edgecolor='k',
                                            facecolor='none')
        ax1.add_feature(china_map, alpha=0.8, linewidth=0.5)
        qq1 = ax1.contourf(longitude, latitude, data1, extend='both', levels=common_levels,norm=norm, transform=ccrs.PlateCarree(),
                           cmap=color)
        cbar=fig1.colorbar(qq1, ax=[ax1],orientation="vertical",shrink=0.5, pad=0.1)
        cbar.outline.set_visible(False)
        plt.savefig(OutputFile_P+F"{Threshold[i]}_NoGrid.png", dpi=800,bbox_inches='tight')
        plt.show()
def Spatial_Statisstical(OutputFile_P,Lable,data,j,color):
    if (j==4 or j ==7):
        CE_Frequency_Map= data
    else:
        CE_Frequency_Map = data.mean(dim='time')
    data_Variables = list(CE_Frequency_Map.data_vars.keys())
    if j==5 or j ==6:
        data_var = sum([CE_Frequency_Map[i] for i in data_Variables]) / len(list(CE_Frequency_Map.data_vars.keys()))*100
    else:
        data_var = sum([CE_Frequency_Map[i] for i in data_Variables]) / len(list(CE_Frequency_Map.data_vars.keys()))
    vmin = np.nanmin(data_var)
    vmax = np.nanmax(data_var)
    if j ==5 or j ==6:
        common_levels = np.linspace(vmin, vmax, 9)
        bins = list(zip(common_levels, common_levels[1:]))
    elif j==7:
        common_levels = np.linspace(vmin, vmax, 9)
        bins = list(zip(common_levels, common_levels[1:]))
    elif j==4:
        common_levels = np.array([vmin, 1,2,3,5,10,20,40,vmax])
        bins = list(zip(common_levels, common_levels[1:]))
    else:
        common_levels = np.linspace(vmin, vmax, 10).astype(int)
        bins = list(zip(common_levels, common_levels[1:]))
    Threshold = np.array([0.7, 0.8, 0.9])
    for i in range(len(Threshold)):
        data = data_var.sel(quantile=Threshold[i])
        counts = []
        for lo, hi in bins:
            # 使用numpy.where获取处于区间内的数据索引
            mask = np.where((data >= lo) & (data < hi))
            # 计算该区间内的数据数量
            counts.append(mask[0].shape[0])
        plt.figure(figsize=(5, 5))
        plt.pie(counts, colors=color,  pctdistance=0.85,startangle=90, wedgeprops={'edgecolor': 'black', 'linewidth': 0.3})
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        plt.savefig(OutputFile_P + F"{Lable}_{Threshold[i]}.png", dpi=800, bbox_inches='tight')
        plt.show()

color_list_F=['#FFFFE7','#FFFFB5','#FFFF83','#FFC340','#FF8607','#D45400','#961700','#580000','#260000']
color_list_D =['#FFFFE7','#FFFFB5','#FFFF83','#FFC340','#FF8607','#D45400','#961700','#580000','#260000']
color_list_SD =['#DDE318','#90D743','#50C46A','#25AC82','#21918C','#2B758E','#375A8C','#443983','#481668']
color_list_ED =['#DDE318','#90D743','#50C46A','#25AC82','#21918C','#2B758E','#375A8C','#443983','#481668']
color_list_RT =[ '#FFFF85','#FFAD2D','#FF8303','#DC5D00','#A82900','#700000','#380000','#0E0000']
color_list_CP = ['#FFFF8F','#FDE725','#BEDF25', '#7AD151', '#22A884', '#2A788E', '#414487', '#440154']
color_list_CT = ['#FFFF8F','#FDE725','#BEDF25', '#7AD151', '#22A884', '#2A788E', '#414487', '#440154']
color_list_LMF = ['#967973', '#D5CA83', '#F9F896', '#B3EF8A', '#1BD16B', '#00AEC5', '#1F59BF', '#440154']
color_list_C = ['#FFFF8F','#FDE725','#BEDF25', '#7AD151', '#22A884', '#2A788E', '#414487', '#440154']
color_list_Array=[color_list_F,color_list_D,color_list_SD,color_list_ED,color_list_RT,color_list_CP,color_list_CT,color_list_LMF]
rain_colormap_C = colors.ListedColormap(color_list_C)
rain_colormap_LMF = colors.ListedColormap(color_list_LMF)

colors_levels=['afmhot_r','afmhot_r','viridis_r','viridis_r','afmhot_r',rain_colormap_C,rain_colormap_C,rain_colormap_LMF]
shapefile="../Data/dazhou.shp"
Labels=['CE_TPF (times)','CE_TPD (days)','CE_TPSD (date)', 'CE_TPED (date)','CE_TPRT (yrs)','CE_TPCP (%)','CE_TPCT (%)','CE_TPLMF']
Variable=['CE_Frequency','CE_Duration','CE_Start','CE_End','CE_Return','CE_Ratio_PR','CE_Ratio_Tem','CE_LMF']
FileType_Array=[i+'_Historical.nc'for i in ['CE_Frequency','CE_Duration','CE_Start','CE_End','CE_Return','CE_Ratio_PR','CE_Ratio_Tem','CE_LMF']]

for j in range(len(FileType_Array)):
    FileType=FileType_Array[j]
    OutputFile_P=f"F:\Experiments\Data_Processing\Results\Analysis\ERA5\Table\{Variable[j]}_Spatial_Average"
    Path="F:\Experiments\Data_Processing\Results"
    FilePath_A= [Path+F"\\{i}\\{FileType}" for i in FilePath]
    datasets = [xr.open_dataset(p) for p in FilePath_A]
    for i in range(len(datasets)):
        if (j==4 or j ==7):
            print(2)
        else:
            datasets[i]['time'] = datasets[i]['time'].astype('datetime64[ns]')
        datasets[i]=datasets[i].rename({Variable[j]:FileType.split('.')[0]+"_"+FilePath[i]})
    combined_datasets = xr.merge(datasets)
    CE_Frequency_Curves=Spatial_By_Average(shapefile,OutputFile_P,combined_datasets,j,colors_levels[j])
    CE_Frequency_Curves=Spatial_Statisstical(OutputFile_P,Variable[j],combined_datasets,j,color_list_Array[j])






