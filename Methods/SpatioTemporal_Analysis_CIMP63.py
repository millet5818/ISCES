"""
绘制 Historical,126,245,585 年线性增长图,以 0.7为例
"""

import xarray as xr
from pylab import *
from scipy import stats
mpl.rcParams["font.sans-serif"] = ["Times New Roman"] # TODO  Windows_Functions
mpl.rcParams['font.size'] = 24# 设置全局字体大小

def Spatial_By_Average_Model(data,j):
    CE_Frequency_Map = data.mean(dim=['lat','lon'])
    CE_Frequency_Map_array=CE_Frequency_Map.to_array()
    data_Variables = list(CE_Frequency_Map.data_vars.keys())
    # 计算置信区间
    alpha = 0.95  # 置信区间的置信水平
    n = len(data_Variables)  # 数据点的数量
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


FilePath=['CE_ACCESS-CM2','CE_BCC-CSM2-MR','CE_CanESM5','CE_CMCC-ESM2','CE_GFDL-ESM4','CE_IPSL-CM6A-LR','CE_MIROC6','CE_NESM3']
Mode=['Historical','126','245','585']
shapefile="../Data/dazhou.shp"
Labels=['CE_TPF (times)']
Variable=['CE_Frequency']
Labels_Model=['Historical(1950-2014)','SSP1-2.6(2015-2100)','SSP2-4.5(2015-2100)','SSP2-4.5(2015-2100)']
for v in range(len(Variable)):
    OutputFile_P = f"E:\CE_DATA\Data_Processing\{Variable[v]}_"
    CE_Mean_All=[]
    CE_Low_All=[]
    CE_High_All=[]
    for m in range(len(Mode)):
        print(Mode[m])
        FileType=Variable[v]+f'_{Mode[m]}.nc'
        Path = "F:\Results"
        FilePath_A = [Path + F"\\{i}\\{FileType}" for i in FilePath]
        datasets = [xr.open_dataset(p) for p in FilePath_A]
        for i in range(len(datasets)):
            datasets[i]['time'] = datasets[i]['time'].astype('datetime64[ns]')
            datasets[i] = datasets[i].rename({Variable[v]: FileType.split('.')[0] + "_" + FilePath[i]})
        combined_datasets = xr.merge(datasets)
        data_mean,ci_low,ci_high = Spatial_By_Average_Model(combined_datasets,v)
        CE_Mean_All.append(data_mean.rename(Mode[m]))
        CE_Low_All.append(ci_low.rename(Mode[m]))
        CE_High_All.append(ci_high.rename(Mode[m]))
    # plt.xticks(np.arange(1950, 2110, 20))
    Colors=['#000000','#3685BC','#548235','#BA5058']
    Shadow_Colors=['gray','#A2CAE2','#97BDAF','#ECD9D0']
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
        handles = []
        labels = []
        for mm in range(len(Mode)):
            plt.plot(CE_Mean_All[mm].time.dt.year, CE_Mean_All[mm].sel(quantile=Threshold[t]), color=Colors[mm])
            plt.fill_between(CE_Mean_All[mm].time.dt.year, CE_Low_All[mm].sel(quantile=Threshold[t]), CE_High_All[mm].sel(quantile=Threshold[t]),facecolor=Shadow_Colors[mm], alpha=0.3)
            handles.append(Line2D([0], [0], color=Colors[mm], lw=2))
            labels.append(f'{[Labels_Model[mm]]}')
        ax.legend(handles, labels, framealpha=0, edgecolor='none')
        plt.savefig(OutputFile_P + F"_{Threshold[t]}.png", dpi=800, bbox_inches='tight')
        plt.show()
