import xarray as xr
import numpy as np
from copulas.bivariate import Clayton,Gumbel,Frank
from copulas.multivariate import GaussianMultivariate
import xclim
from copulas.visualization import compare_3d,scatter_3d
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.distributions.empirical_distribution import ECDF

FileNameList_Pr="F:\Datasets\CMIP6\CMIP6_0.15_Binear\Pr\\NESM3\\Historical\\pr_day_NESM3_historical_r1i1p1f1_gn_1952.nc"
FileNameList_Tem="F:\Datasets\CMIP6\CMIP6_0.15_Binear\Max_Tem\\NESM3\\Historical\\tasmax_day_NESM3_historical_r1i1p1f1_gn_1952.nc"
# Pr_theshold= xr.open_dataset("E:\CE_DATA\Data_Processing\Process_Results\Threshold\Pr\\NESM3\\Historical_Threshold_Pr.nc").Threshold
# Tem_theshold = xr.open_dataset("E:\CE_DATA\Data_Processing\Process_Results\Threshold\Tem\\NESM3\\Historical_Threshold_Tem.nc").Threshold


File_Pr=xr.open_dataset(FileNameList_Pr)
File_Tem=xr.open_dataset(FileNameList_Tem)
File_Pr['pr'] = xclim.core.units.convert_units_to(File_Pr['pr'], "mm/d")
File_Tem['tasmax'] = xclim.core.units.convert_units_to(File_Tem['tasmax'], "degC")

# data1 = pd.Series(File_Pr['pr'][:,29,749], name='A')
# data2 = pd.Series(File_Tem['tasmax'][:,29,749], name='B')
# # 使用pd.concat()合并为一个DataFrame的两列
# df = pd.concat([data1, data2], axis=1)


scaler_temp = StandardScaler()
scaler_precip = StandardScaler()
temperature_series_normalized = scaler_temp.fit_transform(File_Tem['tasmax'][:,29,749].to_numpy().reshape(-1, 1))
precipitation_series_normalized = scaler_precip.fit_transform(File_Pr['pr'][:,29,749].to_numpy().reshape(-1, 1))
data = np.hstack((temperature_series_normalized, precipitation_series_normalized))
temperature_threshold = 1  # 标准化后的高温阈值
precipitation_threshold = 1 # 标准化后的强降雨阈值


# 建立Copula模型
copula = GaussianMultivariate()
copula.fit(data)
# 计算联合概率
joint_probability = copula.cumulative_distribution(np.asarray([temperature_threshold, precipitation_threshold]))


# 计算联合重现期
joint_return_period = 1 / joint_probability

# 计算同现重现期
temp_marginal_cdf = copula.marginals[0].cdf([temperature_threshold])[0]
precip_marginal_cdf = copula.marginals[1].cdf([precipitation_threshold])[0]
concurrent_probability = temp_marginal_cdf * precip_marginal_cdf
concurrent_return_period = 1 / concurrent_probability

# 绘制联合概率密度图
plt.figure(figsize=(10, 8))
plt.scatter(File_Tem['tasmax'][:,29,749], File_Pr['pr'][:,29,749], alpha=0.5, label='Data points')
plt.axhline(y=scaler_precip.inverse_transform([[precipitation_threshold]])[0][0], color='r', linestyle='--', label='Precipitation Threshold')
plt.axvline(x=scaler_temp.inverse_transform([[temperature_threshold]])[0][0], color='b', linestyle='--', label='Temperature Threshold')
plt.xlabel('Temperature')
plt.ylabel('Precipitation')
plt.title('Joint Distribution of Temperature and Precipitation')
plt.legend()
plt.show()

# 绘制边缘概率密度图
plt.figure(figsize=(10, 8))
plt.hist(File_Tem['tasmax'][:,29,749], bins=50, alpha=0.5, label='Temperature')
plt.hist(File_Pr['pr'][:,29,749], bins=50, alpha=0.5, label='Precipitation')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Marginal Distributions of Temperature and Precipitation')
plt.legend()
plt.show()

# 绘制联合重现期和同现重现期
plt.figure(figsize=(10, 8))
plt.bar(['Joint Return Period', 'Concurrent Return Period'], [joint_return_period, concurrent_return_period], color=['blue', 'orange'])
plt.ylabel('Return Period (years)')
plt.title('Joint and Concurrent Return Periods')
plt.show()








