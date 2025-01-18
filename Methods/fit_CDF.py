import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from xclim.indices.generic import threshold_count,compare
from xclim.indices import run_length as rl
import xarray
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = ["Times New Roman"] # TODO  Windows_Functions
mpl.rcParams['font.size'] = 28# 设置全局字体大小

def cdf(data,x):
    return np.sum(data<=x)/len(data)


df=pd.read_csv('D:\zhaozheng\projects\Global Risk\SBAS_INSAR\ZZ\Deformation\Deformation_all_Indicators_Fileter1.csv')

#
#
# x=df['Duration']
# y=2.2103+0.5544*x
#
# fig1 = plt.figure(figsize=(12, 9))
# plt.scatter(df['Duration'], df['Frequency'], s=180,facecolors='none',edgecolors='b',label='Landslides')
#
# plt.plot(x, y, linewidth=5,color='r',label='Best fitted line')
# plt.xlabel('Duration (day)')
# plt.ylabel('Frequency (time)')
# plt.legend()
# fig1.savefig("D:\zhaozheng\projects\Global Risk\金沙江易发性\Figures\时间概率\Fit_LINE.png", dpi=800, bbox_inches='tight')

data=[7.2002,
   14.4077,
   12.7444,
   11.6356,
   13.2988,
   14.4077,
   13.8533,
   12.7444,
   12.1900,
    8.3090,
    7.2002,
   14.9621,
   14.9621,
   12.1900,
   14.4077,
   12.7444,
   12.1900,
   13.2988,
   13.8533,
   15.5165,
   12.1900,
   11.6356,
   13.2988,
   11.6356,
    7.7546,
   10.5267,
   13.8533,
   13.8533,
   11.6356,
   12.1900,
   12.1900,
   13.2988,
   13.2988,
   12.7444,
   11.0811,
   13.8533,
   12.7444,
    3.3192,
    6.0913,
   12.1900,
    6.6457,
   13.8533,
   10.5267,
    8.3090,
   13.8533,
    9.9723,
   12.7444,
   10.5267,
    9.4179,
    7.7546,
   12.1900,
   10.5267,
    9.4179,
    4.9825,
    2.7648,
    9.9723,
    7.2002,
   10.5267,
   11.6356,
   12.1900,
    8.8634,
    8.3090,
    6.0913,
   12.7444,
    8.8634,
    6.6457,
    8.8634,
    6.6457,
   11.6356,
   10.5267,
    7.2002,
    8.3090,
    4.9825,
   10.5267,
    9.9723,
    5.5369,
    9.9723,
    6.6457,
    6.0913,
    6.6457,
    9.4179]
x_values = np.sort(data)
y_values = [cdf(data, x) for x in x_values]
import matplotlib.pyplot as plt
from scipy.stats import norm

# 使用正态分布进行曲线拟合
params = norm.fit(data)
pdf_values = norm.cdf(x_values, *params)




fig1 = plt.figure(figsize=(12, 9))
# 绘制拟合曲线
plt.scatter(x_values, y_values, s=180,facecolors='none',edgecolors='b',label='Landslides')
plt.plot(x_values, pdf_values, linewidth=5,color='r',label='Fitted CDF')
plt.xlabel('Frequency (time)')
plt.ylabel('Temporal probability')
plt.legend()
fig1.savefig("D:\zhaozheng\projects\Global Risk\金沙江易发性\Figures\时间概率\CDF.png", dpi=800, bbox_inches='tight')
