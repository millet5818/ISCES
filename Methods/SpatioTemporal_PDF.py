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

FilePath=['CE_ACCESS-CM2','CE_BCC-CSM2-MR','CE_CanESM5','CE_CMCC-ESM2','CE_GFDL-ESM4','CE_IPSL-CM6A-LR','CE_MIROC6','CE_NESM3']
Mode=['Historical','126','245','585']
Colors = ['Gray', 'green', 'orange', 'red']
shapefile="../Data/dazhou.shp"
Labels=['CE_TPF (times)','CE_TPD (days)','CE_TPSD (date)', 'CE_TPED (date)','CE_TPRT (yrs)','CE_TPCP (%)','CE_TPCT (%)','CE_TPLMF']
Variable=['CE_Frequency','CE_Duration','CE_Start','CE_End','CE_Return','CE_Ratio_PR','CE_Ratio_Tem','CE_LMF']
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
    combined_datasets_ALL_Array=combined_datasets_ALL.to_array()
    Threshold=[0.7,0.8,0.9]
    for tt in range(len(Threshold)):
        value_ddd_A=[]
        for mmm in range(len(Mode)):
            dddd=combined_datasets_ALL_Array[mmm,:,:,tt].to_numpy().flatten()
            value_ddd=dddd[~np.isnan(dddd)]
            value_ddd_A.append(value_ddd)
            # df = pd.DataFrame({Mode[mmm]:value_ddd})
            # df.to_csv(OutputFile_P+F'{Threshold[tt]}_{Mode[mmm]}.csv', index=False)
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 9)  #
        sns.kdeplot(value_ddd_A[0], shade=True, color=Colors[0], alpha=.2,linewidth=3)
        sns.kdeplot(value_ddd_A[1], shade=True, color=Colors[1], alpha=.2,linewidth=3)
        sns.kdeplot(value_ddd_A[2], shade=True, color=Colors[2], alpha=.2,linewidth=3)
        sns.kdeplot(value_ddd_A[3], shade=True, color=Colors[3], alpha=.2,linewidth=3)
        # sns.rugplot(value_ddd_A[2], height=-0.05, color='g', alpha=0.2,clip_on=False)
        plt.savefig(OutputFile_P + F"_{Threshold[tt]}.png", dpi=800, bbox_inches='tight')
        plt.show()
