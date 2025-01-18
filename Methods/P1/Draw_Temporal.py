"""
绘制 Historical,126,245,585 年线性增长图,以 0.7为例
"""

import xarray as xr
from pylab import *
import geopandas as gpd
from scipy import stats
from shapely.geometry import mapping
mpl.rcParams["font.sans-serif"] = ["Times New Roman"] # TODO  Windows_Functions
mpl.rcParams['font.size'] = 24# 设置全局字体大小

def Spatial_By_Average_Model(data,j):
    CE_Frequency_Map = data.mean(dim=['lat','lon'])
    CE_Frequency_Map_array=CE_Frequency_Map.to_array()
    data_Variables = list(CE_Frequency_Map.data_vars.keys())
    # 计算置信区间
    alpha = 0.95  # 置信区间的置信水平
    # n = len(data_Variables)
    n=8# 数据点的数量
    t_ci = stats.t.ppf(1 - (1 - alpha / 2), n - 1)

    if j==4 or j ==5:
        # data_mean = sum([CE_Frequency_Map[i] for i in data_Variables]) / len(list(CE_Frequency_Map.data_vars.keys()))*100
        data_mean=np.mean(CE_Frequency_Map_array,axis=0)*100
        data_std = np.mean(CE_Frequency_Map_array, axis=0)*100
        ci_low = data_mean - t_ci * (data_std / np.sqrt(n))
        ci_high = data_mean + t_ci * (data_std /np.sqrt(n))
    else:
        # data_mean = sum([CE_Frequency_Map[i] for i in data_Variables]) / len(list(CE_Frequency_Map.data_vars.keys()))
        data_mean = np.mean(CE_Frequency_Map_array, axis=0)
        data_std = np.mean(CE_Frequency_Map_array, axis=0)
        ci_low = data_mean - t_ci * (data_std / np.sqrt(n))
        ci_high = data_mean + t_ci * (data_std / np.sqrt(n))

    return data_mean,ci_low,ci_high



Mode=['Historical','585']
CE_Type=['CE','CES','CET','CETS']
shapefile="../../Data/China_All.shp"
Variable=['Frequency','Amplitude','Duration','Start','End']
Labels=['Frequency(times)','Amplitude (days)','Duration (days)','Start Date (doy)','End date (doy)']
Labels_Model=['CEoST','CEpS','CEpT','CEpST','CEoST','CEpS','CEpT','CEpST']
for v in range(len(Variable)):
    # OutputFile_P = f"E:\CE_DATA\Data_Processing\{Variable[v]}_"
    OutputFile_P = f"D:\zhaozheng\projects\open software\Figures\Lines\\{Variable[v]}_"
    CE_Mean_All=[]
    CE_Low_All=[]
    CE_High_All=[]
    for m in range(len(Mode)):
        for ty in range(len(CE_Type)):
            FileType=Variable[v]+f'_{Mode[m]}.nc'
            Path = F"E:\CE_DATA\Data_Processing\Average_Mode"
            FilePath_A = Path + f"\\{CE_Type[ty]}\\CMIP6\\{CE_Type[ty]}_"+f"{Variable[v]}_{Mode[m]}.nc"
            datasets = xr.open_dataset(FilePath_A)
            gdf1 = gpd.read_file(shapefile)
            datasets.rio.write_crs("epsg:4326",inplace=True)
            datasets.rio.set_spatial_dims(x_dim="lon",y_dim="lat",inplace=True)
            datasets=datasets.rio.clip(gdf1.geometry.apply(mapping),gdf1.crs,drop=True)
            data_mean,ci_low,ci_high = Spatial_By_Average_Model(datasets,1)
            CE_Mean_All.append(data_mean.rename(Mode[m]))
            CE_Low_All.append(ci_low.rename(Mode[m]))
            CE_High_All.append(ci_high.rename(Mode[m]))
    # plt.xticks(np.arange(1950, 2110, 20))
    Colors=['#000000','#3685BC','#548235','#BA5058','#000000','#3685BC','#548235','#BA5058']
    Shadow_Colors=['gray','#A2CAE2','#97BDAF','#ECD9D0','gray','#A2CAE2','#97BDAF','#ECD9D0']
    Threshold = np.array([0.7, 0.8, 0.9])
    for t in range(len(Threshold)):
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 9)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)  #
        plt.tick_params(which='both', width=4, length=10)
        plt.rc('font', weight='bold')
        plt.xticks([1950, 1980, 2010, 2040, 2070, 2100])
        ax.set_ylabel(Labels[v])
        # 在 y=0.5 的地方添加一条竖线
        plt.axvline(x=2014, color='#A6A6A6', linestyle='--')
        # 添加阴影区域，表示 y=0.5 的准确性
        plt.axvspan(1950, 2014, facecolor='#DBDBDB', alpha=0.5)
        plt.axvspan(2014, 2100, facecolor='#FFF8C2', alpha=0.5)
        handles = []
        labels = []
        for mm in range(8):
            plt.plot(CE_Mean_All[mm].time.dt.year, CE_Mean_All[mm].sel(quantile=Threshold[t]), color=Colors[mm])
            plt.fill_between(CE_Mean_All[mm].time.dt.year, CE_Low_All[mm].sel(quantile=Threshold[t]), CE_High_All[mm].sel(quantile=Threshold[t]),facecolor=Shadow_Colors[mm], alpha=0.3)
            if mm<4:
                handles.append(Line2D([0], [0], color=Colors[mm], lw=2))
                labels.append(f'{Labels_Model[mm]}')
        ax.legend(handles, labels, framealpha=0, edgecolor='none')
        plt.savefig(OutputFile_P + F"{Variable[v]}_{Threshold[t]}.png", dpi=800, bbox_inches='tight')

