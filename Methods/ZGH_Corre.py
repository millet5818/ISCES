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
import scipy.stats



def cal_cof():
    defo_name="D:\zhaozheng\projects\全球风险计算\SBAS_INSAR\ZGH\zgh_2024\Rock_avalanche_All.csv"
    data_defo=pd.read_csv(defo_name).values
    num_array=np.arange(0,15,1)
    corre_array=np.ones((15,int((data_defo.shape[1]-2)/2+1)))
    for ni in range(num_array.shape[0]):
        print(ni)
        filename=f"E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZGH\Typical_Landslides\ZGH-ZHIHOU\RA_Daily_Tem_max{ni}.csv"
        data_pre=pd.read_csv(filename,header=None).values
        data_cor=[]
        for i in range(int((data_defo.shape[1]-2)/2)):
            corr_single=scipy.stats.pearsonr(data_defo[1:,i+int((data_defo.shape[1]-2)/2)+2],data_pre[1:,i])[0]
            data_cor.append(corr_single)
        data_cor.append(np.mean(np.abs(data_cor)))
        corre_array[ni,:]=data_cor
    np.savetxt("E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZGH\Typical_Landslides\RA_Tem_Cof.csv", corre_array, delimiter=',')


plt.figure(figsize=(10, 7))
sns.set(font_scale=1.2)  # 设置字体比例
filename="E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZGH\Typical_Landslides\ZGH-ZHIHOU\COF\Tem_Cof.csv"
df=pd.read_csv(filename)
df_new=pd.DataFrame(df.values[:,1:],columns=['RA',	'RF',	'RG',	'RS',	'RSD'], index=['1', '2', '3', '4', '5','6', '7', '8', '9', '10','11', '12', '13', '14', '15'])
heatmap = sns.heatmap(df_new, annot=True, cmap='YlOrRd', fmt='.2f')
plt.ylabel('Time lag (day)')
plt.savefig("E:\CE_DATA\Data_Processing\Average_Mode\Cut_ZGH\Typical_Landslides\ZGH-ZHIHOU\Tem_COF.png", dpi=800, bbox_inches='tight')
plt.show()
