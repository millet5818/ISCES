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
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from matplotlib.ticker  import MaxNLocator
import rioxarray
import matplotlib.gridspec as gridspec
from shapely.geometry import mapping
mpl.rcParams["font.sans-serif"] = ["Times New Roman"] # TODO  Windows_Functions
mpl.rcParams['font.size'] = 24# 设置全局字体大小





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


def getColor(j,vmin,vmax,color):
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
        common_levels = np.linspace(vmin, vmax, 12).astype(int)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    return common_levels, norm

def DrawSpatial_Model(shapefile,OutputFile_P,data,j,color):
    print(1)
    # Threshold = np.array([0.7, 0.8, 0.9])
    Threshold = np.array([0.7])
    data_Variables = list(data.data_vars.keys())
    for i in range(len(Threshold)):
        for m in range(len(data_Variables)):
            time_period=[['1981','2010'],['2026','2055'],['2071','2100']]
            Period_Name=["Historical","Near","Far"]
            Vmin=[]
            Vmax=[]
            for ti in time_period:
                data_single1=data[data_Variables[m]].sel(quantile=Threshold[i]).sel(time=slice(ti[0],ti[1])).mean(dim='time')
            # data_single1=data[data_Variables[m]].sel(quantile=Threshold[i]).sel(time=slice('1950','2100')).mean(dim='time')
                Vmin.append(np.min(data_single1.to_pandas()))
                Vmax.append(np.max(data_single1.to_pandas()))
            common_levels,norm=getColor(j,np.min(Vmin),np.max(Vmax),color)
            for ti in time_period:
                data_single=data[data_Variables[m]].sel(quantile=Threshold[i]).sel(time=slice(ti[0],ti[1])).mean(dim='time')
                longitude = data_single.lon
                latitude = data_single.lat
                gdf = gpd.read_file(shapefile)
                maskregion1 = data_single.salem.roi(shape=gdf)
                data1 = maskregion1.values
                fig1 = plt.figure(figsize=(16, 9))
                # gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[2, 1])
                proj = ccrs.PlateCarree()
                ax1 = fig1.add_subplot(111, projection=ccrs.PlateCarree())
                # ax1.imshow(China_tif_im,extent=[73, 135, 3, 53],transform=proj) #,
                # ax1.imshow(China_tif_im,extent=[-180,180,-90,90],transform=proj) #,
                china_map = cfeature.ShapelyFeature(shpreader.Reader(shapefile).geometries(), proj, edgecolor='black',
                                                    facecolor='none')
                ax1.add_feature(china_map, alpha=0.6, linewidth=1.5)
                china_zonation = cfeature.ShapelyFeature(shpreader.Reader(shapefile_climate).geometries(), proj, edgecolor='black',
                                                    facecolor='none')
                ax1.add_feature(china_zonation, alpha=0.7, linewidth=0.7)
                ax1.set_extent([73, 135, 18, 55], crs=ccrs.PlateCarree())
                gl = ax1.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,linewidth=0.2)
                gl.xlabels_top = False
                gl.ylabels_right = False
                qq1 = ax1.contourf(longitude, latitude, data1, extend='both', levels=common_levels, norm=norm,
                                   transform=ccrs.PlateCarree(),cmap=color)
                cbar = fig1.colorbar(qq1, ax=[ax1], orientation="horizontal", shrink=0.5, pad=0.1)
                sub_ax = fig1.add_axes([0.649, 0.308, 0.15, 0.15],projection=ccrs.PlateCarree())
                sub_ax.add_feature(china_map, alpha=0.6, linewidth=1.5)
                sub_ax.contourf(longitude, latitude, data1, extend='both', levels=common_levels, norm=norm,
                                   transform=ccrs.PlateCarree(),cmap=color)
                sub_ax.set_extent([105, 125, 0, 25], crs=ccrs.PlateCarree())
                plt.savefig(OutputFile_P + F"{data_Variables[m]}_{ti[0]}_{ti[1]}_{Threshold[i]}.png", dpi=800, bbox_inches='tight')
                # plt.show()

Variable=['Frequency','Amplitude','Duration','Start','End']
Mode=['Historical','585']
shapefile="../../Data/China_All.shp"
CE_Type=['CE','CES','CET','CETS']
# shapefile="../../Data/China_Land.shp"
shapefile_add="../../Data/NineLine.shp"
# China_Tif="D:\zhaozheng\projects\ExtractionCompoundEvents\ISCES\Data\Extract_tif764.tif"
shapefile_climate="../../Data/Chinese_climate1.shp"
color_list_C = ['#FFFF8F','#FDE725','#BEDF25', '#7AD151', '#22A884', '#2A788E', '#414487', '#440154']
color_list_LMF = ['#5E4FA1', '#D5CA83', '#F9F896', '#B3EF8A', '#1BD16B', '#00AEC5', '#1F59BF', '#440154']
color_list_Frequency = ['#5E4FA1', '#5C79B8', '#5091C3', '#7EC4AB', '#B5D7AA', '#E5E89E', '#FFF8C2', '#FDE296','#F7B370','#EE7B51','#D9545D','#9D0142']
rain_colormap_C = colors.ListedColormap(color_list_C)
rain_colormap_LMF = colors.ListedColormap(color_list_LMF)
rain_colormap_Frequency = colors.ListedColormap(color_list_Frequency)
colors_levels=['afmhot_r','afmhot_r','viridis_r','viridis_r','afmhot_r',rain_colormap_C,rain_colormap_C,rain_colormap_LMF]

# China_tif_im=plt.imread(China_Tif)

for v in range(len(Variable)):
    for ty in range(len(CE_Type)):
        OutputFile_P = f"D:\zhaozheng\projects\open software\Figures\\test2\{Variable[v]}_"
        CE_All=[]
        Path = F"E:\CE_DATA\Data_Processing\Average_Mode"
        FilePath_A = [Path + f"\\{CE_Type[ty]}\\CMIP6\\{CE_Type[ty]}_"+f"{Variable[v]}_{i}.nc" for i in Mode]
        datasets = [xr.open_dataset(p) for p in FilePath_A]
        combined_datasets = xr.merge(datasets)
        gdf1 = gpd.read_file(shapefile)
        combined_datasets.rio.write_crs("epsg:4326",inplace=True)
        combined_datasets.rio.set_spatial_dims(x_dim="lon",y_dim="lat",inplace=True)
        combined_datasets=combined_datasets.rio.clip(gdf1.geometry.apply(mapping),gdf1.crs,drop=True)
        # combined_datasets.to_netcdf("D:\zhaozheng\projects\open software\Figures\dsds.nc")
        DrawSpatial_Model(shapefile,OutputFile_P,combined_datasets,v,rain_colormap_Frequency)
