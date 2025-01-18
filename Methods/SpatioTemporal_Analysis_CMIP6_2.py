"""
绘制 Historical,126,245,585 空间分布图,以 0.7为例
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
import jenkspy
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from matplotlib.ticker  import MaxNLocator
mpl.rcParams["font.sans-serif"] = ["Times New Roman"] # TODO  Windows_Functions
mpl.rcParams['font.size'] = 16# 设置全局字体大小


def Spatial_By_Average_Model(data,j):
    if (j==4 or j ==7):
        CE_Frequency_Map= data
    else:
        CE_Frequency_Map = data.mean(dim='time')
    data_Variables = list(CE_Frequency_Map.data_vars.keys())
    if j==5 or j ==6:
        data_var = sum([CE_Frequency_Map[i] for i in data_Variables]) / len(list(CE_Frequency_Map.data_vars.keys()))*100
    else:
        data_var = sum([CE_Frequency_Map[i] for i in data_Variables]) / len(list(CE_Frequency_Map.data_vars.keys()))
    return data_var


def DrawSpatial_Model(shapefile,OutputFile_P,data,j,color):
    print(1)
    data_Variables = list(data.data_vars.keys())
    vmin_all = data.min(dim=['lon','lat'])
    vmax_all = data.max(dim=['lon','lat'])
    Threshold = np.array([0.7, 0.8, 0.9])
    for i in range(len(Threshold)):
        print(Threshold[i])
        vmin=np.min([vmin_all.sel(quantile=Threshold[i])[t] for t in data_Variables])
        vmax=np.max([vmax_all.sel(quantile=Threshold[i])[t] for t in data_Variables])
        if j == 5 or j == 6:
            common_levels = np.linspace(vmin, vmax, 9)
            norm = colors.BoundaryNorm(common_levels, color.N)
        elif j == 7:
            common_levels = np.linspace(vmin, vmax, 9)
            norm = colors.BoundaryNorm(common_levels, color.N)
        elif j == 4:
            common_levels = np.array([vmin, 1, 2, 3, 5, 10, 20, 40, vmax])
            norm = colors.SymLogNorm(linthresh=0.5, linscale=0.5, vmin=vmin, vmax=vmax, base=20)
        else:
            common_levels = np.linspace(vmin, vmax, 10).astype(int)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

        for m in range(len(data_Variables)):
            data_single=data[data_Variables[m]].sel(quantile=Threshold[i])
            longitude = data_single.lon
            latitude = data_single.lat
            gdf = gpd.read_file(shapefile)
            maskregion1 = data_single.salem.roi(shape=gdf)
            data1 = maskregion1.values
            fig1 = plt.figure(figsize=(16, 9))
            proj = ccrs.PlateCarree()
            ax1 = fig1.add_subplot(111, projection=ccrs.PlateCarree())
            china_map = cfeature.ShapelyFeature(shpreader.Reader(shapefile).geometries(), proj, edgecolor='k',
                                                facecolor='none')
            ax1.add_feature(china_map, alpha=0.8, linewidth=0.5)
            qq1 = ax1.contourf(longitude, latitude, data1, extend='both', levels=common_levels, norm=norm,
                               transform=ccrs.PlateCarree(),
                               cmap=color)
            cbar = fig1.colorbar(qq1, ax=[ax1], orientation="vertical", shrink=0.5, pad=0.1)
            cbar.outline.set_visible(False)
            plt.savefig(OutputFile_P + F"{data_Variables[m]}_{Threshold[i]}.png", dpi=800, bbox_inches='tight')
            plt.show()

FilePath=['CE_ACCESS-CM2','CE_BCC-CSM2-MR','CE_CanESM5','CE_CMCC-ESM2','CE_GFDL-ESM4','CE_IPSL-CM6A-LR','CE_MIROC6','CE_NESM3']
Mode=['Historical','126','245','585']
shapefile="../Data/dazhou.shp"
Labels=['CE_TPF (times)','CE_TPD (days)','CE_TPSD (date)', 'CE_TPED (date)','CE_TPRT (yrs)','CE_TPCP (%)','CE_TPCT (%)','CE_TPLMF']
Variable=['CE_Frequency','CE_Duration','CE_Start','CE_End','CE_Return','CE_Ratio_PR','CE_Ratio_Tem','CE_LMF']
color_list_C = ['#FFFF8F','#FDE725','#BEDF25', '#7AD151', '#22A884', '#2A788E', '#414487', '#440154']
color_list_LMF = ['#967973', '#D5CA83', '#F9F896', '#B3EF8A', '#1BD16B', '#00AEC5', '#1F59BF', '#440154']
rain_colormap_C = colors.ListedColormap(color_list_C)
rain_colormap_LMF = colors.ListedColormap(color_list_LMF)
colors_levels=['afmhot_r','afmhot_r','viridis_r','viridis_r','afmhot_r',rain_colormap_C,rain_colormap_C,rain_colormap_LMF]

for v in range(len(Variable)):
    print(v)
    OutputFile_P = f"F:\Experiments\Data_Processing\Results\Analysis\ERA5\Table\{Variable[v]}_"
    CE_All=[]
    for m in range(len(Mode)):
        print(Mode[m])
        FileType=Variable[v]+f'_{Mode[m]}.nc'
        Path = "F:\Experiments\Data_Processing\Results"
        FilePath_A = [Path + F"\\{i}\\{FileType}" for i in FilePath]
        datasets = [xr.open_dataset(p) for p in FilePath_A]
        for i in range(len(datasets)):
            if (v == 4 or v == 7):
                print(2)
            else:
                datasets[i]['time'] = datasets[i]['time'].astype('datetime64[ns]')
            datasets[i] = datasets[i].rename({Variable[v]: FileType.split('.')[0] + "_" + FilePath[i]})
        combined_datasets = xr.merge(datasets)
        CE_Frequency_Curves = Spatial_By_Average_Model(combined_datasets, v)
        CE_All.append(CE_Frequency_Curves.rename(Mode[m]))
    combined_datasets_ALL = xr.merge(CE_All)
    DrawSpatial_Model(shapefile,OutputFile_P,combined_datasets_ALL,v,colors_levels[v])
