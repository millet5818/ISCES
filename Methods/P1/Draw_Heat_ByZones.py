
import xarray as xr
from pylab import *
import geopandas as gpd
from scipy import stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import mapping
mpl.rcParams["font.sans-serif"] = ["Times New Roman"] # TODO  Windows_Functions
mpl.rcParams['font.size'] = 16# 设置全局字体大小

Mode=['Historical','585']
CE_Type=['CE','CES','CET','CETS']
shapefile="../../Data/Chinese_climate.shp"
Variable=['Frequency','Amplitude','Duration']
Labels=['Frequency(times)','Amplitude (days)','Duration (days)']
shapefile = gpd.read_file(shapefile)

for v in range(len(Variable)):
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 9)
    OutputFile_P = f"D:\zhaozheng\projects\open software\Figures\{Variable[v]}_"
    Path = F"E:\CE_DATA\Data_Processing\Average_Mode"
    FilePath_A = [Path + "\\CETS\\CMIP6\\CETS_"+f"{Variable[v]}_{i}.nc" for i in Mode]
    datasets = [xr.open_dataset(p) for p in FilePath_A]
    combined_datasets = xr.merge(datasets)
    combined_datasets.rio.write_crs("epsg:4326",inplace=True)
    combined_datasets.rio.set_spatial_dims(x_dim="lon",y_dim="lat",inplace=True)
    data_heatmap=np.zeros((7,combined_datasets.time.dt.year.shape[0]))
    for shi in range(7):
        Part_Shape=shapefile[shapefile.Climate_ID==shi+1]# 获取部分要素
        Clip_datasets=combined_datasets.rio.clip(Part_Shape.geometry.apply(mapping),Part_Shape.crs,drop=True).sel(quantile=0.7).mean(dim=['lat','lon'])
        data_heatmap[shi,:]=Clip_datasets.to_array().values[0,:]
    # df=pd.DataFrame(data_heatmap,columns=['North China is a humid and semi-humid temperate region','Moist subtropical region in central and south China','Northwest desert area','Northeast humid and subhumid area','Inner Mongolia grassland area', 'Qinghai-Tibetan Plateau', 'tropical region of south China'],rows=combined_datasets.time.dt.year.to_numpy())
    # columns=['North China is a humid and semi-humid temperate region','Moist subtropical region in central and south China','Northwest desert area','Northeast humid and subhumid area','Inner Mongolia grassland area', 'Qinghai-Tibetan Plateau', 'tropical region of south China']
    columns=['HST','MSR','NDA','NHS','IMG', 'QTP', 'TRS']
    Years=combined_datasets.time.dt.year.to_numpy()
    df=pd.DataFrame(data_heatmap,columns=Years,index=columns)
    sns.heatmap(df,cmap='viridis',annot=False,linewidths=0.5)
    plt.savefig(OutputFile_P + F"{Variable[v]}_heatmap.png", dpi=800, bbox_inches='tight')
