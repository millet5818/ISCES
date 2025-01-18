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
from scipy.stats import linregress
from pylab import *
from scipy.stats.mstats import ttest_ind
from sklearn.linear_model import LinearRegression
from matplotlib.ticker  import MaxNLocator
mpl.rcParams["font.sans-serif"] = ["Times New Roman"] # TODO  Windows_Functions
mpl.rcParams['font.size'] = 16# 设置全局字体大小




def Spatial_By_Average_Model(data,j):
    CE_Frequency_Map= data.to_array()
    if j==4 or j ==5:
        data_var = np.mean(CE_Frequency_Map,axis=0)*100
    else:
        data_var = np.mean(CE_Frequency_Map,axis=0)
    return data_var



def  Cal_Trending1(OutputFile_P,OutputFile,data):
    data_var=data[list(data.data_vars.keys())[0]]
    trend = np.zeros((3,data_var.latitude.shape[0], data_var.longitude.shape[0]))
    p_value = np.zeros((3,data_var.latitude.shape[0], data_var.longitude.shape[0]))
    intercept=np.zeros((3))
    r_value=np.zeros((3))
    Threshold=[0.7,0.8,0.9]
    for t in range(len(Threshold)):
        data_var_t=data_var.sel(quantile=Threshold[t])
        for i in range (0,data_var.latitude.shape[0]):
            print(i)
            for j in range (0,data_var.longitude.shape[0]):
                # TODO 什么情况下进行线性回归，同一个像素，百分之六十的年份存在复合事件，则进行趋势检验
                trend[t,i,j], intercept[t], r_value[t], p_value[t,i,j], std_err=stats.linregress(np.arange(1941,2024,1),data_var_t[:,i,j])
    np.save(OutputFile,trend)
    np.save(OutputFile_P,p_value)

def  Cal_Trending(OutputFile_P,OutputFile,data,m):
        # data_var=data[list(data.data_vars.keys())[0]]
        data_var=data
        trend = np.zeros((3,data_var.lat.shape[0], data_var.lon.shape[0]))
        p_value = np.zeros((3,data_var.lat.shape[0], data_var.lon.shape[0]))
        intercept=np.zeros((3))
        r_value=np.zeros((3))
        Threshold=[0.7,0.8,0.9]
        for t in range(len(Threshold)):
            data_var_t=data_var.sel(quantile=Threshold[t])
            for i in range(0,data_var.lat.shape[0]):
                print(i)
                for j in range(0,data_var.lon.shape[0]):
                    # TODO 什么情况下进行线性回归，同一个像素，百分之六十的年份存在复合事件，则进行趋势检验
                    # trend[t,i,j], intercept[t], r_value[t], p_value[t,i,j], std_err=stats.linregress(np.arange(1941,2024,1),data_var_t[:,i,j])
                    trend[t,i,j], intercept[t], r_value[t], p_value[t,i,j], std_err=stats.linregress(data.time.dt.year.to_numpy(),data_var_t[:,i,j])
        np.save(OutputFile+f'{m}.npy',trend)
        np.save(OutputFile_P+f'{m}.npy',p_value)

def Draw_Trennding(shapefile,data,data_trend,data_p,Export_Firgure):
    print(1)
    # TODO 交换顺序
    data_trend = data_trend.transpose(1,2,0)
    data_p = data_p.transpose(1, 2, 0)
    CE_Frequency_Map = data.mean(dim='time')
    CE_Frequency_Map[list(data.data_vars.keys())[0]].values=data_trend
    data_p_Map=CE_Frequency_Map
    data_p_Map[list(data.data_vars.keys())[0]].values=data_p
    vmin = np.nanmin(CE_Frequency_Map[list(data.data_vars.keys())[0]])
    vmax = np.nanmax(CE_Frequency_Map[list(data.data_vars.keys())[0]])
    common_levels = np.linspace(vmin, vmax, 10)
    # norm = colors.Normalize(vmin=vmin, vmax=vmax)
    norm = colors.SymLogNorm(linthresh=0.5, linscale=0.5, vmin=vmin, vmax=vmax, base=20)
    # color_list = ['#967973', '#D5CA83', '#F9F896', '#B3EF8A', '#1BD16B', '#00AEC5', '#1F59BF', '#440154']
    # rain_colormap = colors.ListedColormap(color_list)
    # norm = colors.BoundaryNorm(common_levels, rain_colormap.N)
    Threshold = np.array([0.7, 0.8, 0.9])
    longitude = data.lon
    latitude = data.lat
    gdf = gpd.read_file(shapefile)
    for i in range(len(Threshold)):
        print(i)
        data_single = CE_Frequency_Map[list(data.data_vars.keys())[0]].sel(quantile=Threshold[i])
        maskregion1 = data_single.salem.roi(shape=gdf)
        data1 = maskregion1.values
        data_p = data_p_Map[list(data.data_vars.keys())[0]].sel(quantile=Threshold[i])
        fig1 = plt.figure(figsize=(16, 9))
        proj = ccrs.PlateCarree()
        ax1 = fig1.add_subplot(111, projection=ccrs.PlateCarree())
        china_map = cfeature.ShapelyFeature(shpreader.Reader(shapefile).geometries(), proj, edgecolor='k',
                                            facecolor='none')
        ax1.add_feature(china_map, alpha=0.8, linewidth=0.5)
        qq1 = ax1.contourf(longitude, latitude, data1, extend='both', levels=common_levels,
                           norm=norm, cmap='afmhot_r',transform=ccrs.PlateCarree())  # viridis
        cbar = fig1.colorbar(qq1, ax=[ax1], orientation="vertical", shrink=0.5, pad=0.1)
        c1b = ax1.contourf(longitude, latitude, data_p, [np.min(data_p), 0.05, np.max(data_p)], hatches=['.', None],
                          zorder=1, colors="none", transform=ccrs.PlateCarree())
        cbar.outline.set_visible(False)
        plt.savefig(Export_Firgure + F"{Threshold[i]}_NoGrid.png", dpi=800, bbox_inches='tight')
        plt.show()

# FilePath=['CE_ACCESS-CM2','CE_BCC-CSM2-MR','CE_CanESM5','CE_CMCC-ESM2','CE_GFDL-ESM4','CE_IPSL-CM6A-LR','CE_MIROC6','CE_NESM3']
# Mode=['Historical','126','245','585']
# shapefile="../Data/dazhou.shp"
# Labels=['CE_TPF (times)','CE_TPD (days)','CE_TPSD (date)', 'CE_TPED (date)','CE_TPCP (%)','CE_TPCT (%)']
# Variable=['CE_Frequency','CE_Duration','CE_Start','CE_End','CE_Ratio_PR','CE_Ratio_Tem']
# # for v in range(len(Variable)):
# for v in range(4,5):
#     print(v)
#     OutputFile_P = f"F:\Experiments\Data_Processing\Results\Analysis\ERA5\Table\\{Variable[v]}_"
#     CE_All=[]
#     # for m in range(len(Mode)):
#     for m in range(1,4):
#         print(Mode[m])
#         FileType=Variable[v]+f'_{Mode[m]}.nc'
#         Path = "F:\Experiments\Data_Processing\Results"
#         FilePath_A = [Path + F"\\{i}\\{FileType}" for i in FilePath]
#         datasets = [xr.open_dataset(p) for p in FilePath_A]
#         for i in range(len(datasets)):
#             datasets[i]['time'] = datasets[i]['time'].astype('datetime64[ns]')
#             datasets[i] = datasets[i].rename({Variable[v]: FileType.split('.')[0] + "_" + FilePath[i]})
#         combined_datasets = xr.merge(datasets)
#         CE_Frequency_Curves = Spatial_By_Average_Model(combined_datasets, v)
#         # CE_All.append(CE_Frequency_Curves.rename(Mode[m]))
#         combined_datasets_ALL = CE_Frequency_Curves.rename(Mode[m])
#         OutputFile = f"F:\Experiments\Data_Processing\Results\Analysis\ERA5\Table\趋势分析\CE_Trend_{Variable[v]}_"
#         OutputFile_P = f"F:\Experiments\Data_Processing\Results\Analysis\ERA5\Table\趋势分析\CE_value_{Variable[v]}_"
#         CE_Frequency_Curves = Cal_Trending(OutputFile, OutputFile_P, combined_datasets_ALL,Mode[m])
    # combined_datasets_ALL = xr.merge(CE_All)
    # combined_datasets_ALL = CE_All
    # OutputFile=f"F:\Experiments\Data_Processing\Results\Analysis\ERA5\Table\趋势分析\CE_Trend_{Variable[v]}_"
    # OutputFile_P=f"F:\Experiments\Data_Processing\Results\Analysis\ERA5\Table\趋势分析\CE_value_{Variable[v]}_"
    # CE_Frequency_Curves=Cal_Trending(OutputFile,OutputFile_P,combined_datasets_ALL)

data=xr.open_dataset("E:\CE_DATA\Data_Processing\Process_Results\CETS_CanESM5\CE_Frequency_585.nc")
OutputFile="E:\CE_DATA\Data_Processing\Process_Results\CMIP6\CE_Trend_CE_Ratio_PR_585.npy"
OutputFile_P="E:\CE_DATA\Data_Processing\Process_Results\CMIP6\CE_value_CE_Ratio_PR_585.npy"
Export_Firgure="E:\CE_DATA\Data_Processing\Process_Results\CMIP6\CE_TPCP_trend"
shapefile="../Data/dazhou.shp"
data_trend=np.load(OutputFile)
data_p=np.load(OutputFile)
CE_Frequency_Curves=Draw_Trennding(shapefile,data,data_trend,data_p,Export_Firgure)



