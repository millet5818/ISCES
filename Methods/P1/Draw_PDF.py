"""
历史和未来时期全球年平均复合事件特征的概率密度函数( PDF )
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
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from matplotlib.ticker  import MaxNLocator
import matplotlib.gridspec as gridspec
from shapely.geometry import mapping
mpl.rcParams["font.sans-serif"] = ["Times New Roman"] # TODO  Windows_Functions
mpl.rcParams['font.size'] = 32# 设置全局字体大小

import seaborn as sns


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

Variable=['Frequency','Amplitude','Duration','Start','End']
# Variable=['Amplitude']
CE_Type=['CE','CES','CET','CETS']
Mode=['Historical','126','245','585']
Colors = ['Gray', 'green', 'orange', 'red']
shapefile="../../Data/China_All.shp"
Labels=['CE_TPF (times)','CE_TPD (days)','CE_TPSD (date)', 'CE_TPED (date)']

for v in range(len(Variable)):
    print(v)
    OutputFile_P = f"D:\zhaozheng\projects\open software\Figures\{Variable[v]}_"
    CE_All=[]
    for m in range(len(Mode)):
        print(Mode[m])
        FileType=Variable[v]+f'_{Mode[m]}.nc'
        Path = F"E:\CE_DATA\Data_Processing\Average_Mode"
        FilePath_A = [Path +F"\\{i}\\CMIP6"+ F"\\{i}_{FileType}" for i in CE_Type]
        datasets = [xr.open_dataset(p) for p in FilePath_A]
        combined_datasets = xr.merge(datasets).sel(quantile=0.7).mean(dim='time')
        gdf1 = gpd.read_file(shapefile)
        combined_datasets.rio.write_crs("epsg:4326",inplace=True)
        combined_datasets.rio.set_spatial_dims(x_dim="lon",y_dim="lat",inplace=True)
        combined_datasets=combined_datasets.rio.clip(gdf1.geometry.apply(mapping),gdf1.crs,drop=True)
        combined_datasets_ALL_Array=combined_datasets.to_array()
        value_ddd_A=[]
        for mmm in range(len(CE_Type)):
            dddd=combined_datasets_ALL_Array[mmm,:,:].to_numpy().flatten()
            value_ddd=dddd[~np.isnan(dddd)]
            value_ddd_A.append(value_ddd)
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 9)  #
        sns.kdeplot(value_ddd_A[0], shade=True, color=Colors[0], alpha=.2,linewidth=3)
        sns.kdeplot(value_ddd_A[1], shade=True, color=Colors[1], alpha=.2,linewidth=3)
        sns.kdeplot(value_ddd_A[2], shade=True, color=Colors[2], alpha=.2,linewidth=3)
        sns.kdeplot(value_ddd_A[3], shade=True, color=Colors[3], alpha=.2,linewidth=3)
        # sns.rugplot(value_ddd_A[2], height=-0.05, color='g', alpha=0.2,clip_on=False)
        plt.savefig(OutputFile_P + F"_PDF_{Mode[m]}.png", dpi=800, bbox_inches='tight')
