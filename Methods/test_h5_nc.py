

import xarray as xr

import numpy as np

import glob
import os
#
# for j in range(2,3):
#     print(j)
#
# arr1 = xr.DataArray([[0,0,0],[1,1,0],[0,0,1]],dims=("x","y"))
#
#
# arr2 = xr.DataArray([[0,1,1],[0,1,0],[0,0,1]],dims=("x","y"))
#
#
#
# cond_pr_roll=arr1.rolling(x=3,y=3,center=True,min_periods=1).max()
# # todo 只有一个实现左右采样就可以了，多了就不对了，就架空中间一个了
# cond_tem_roll=arr2.rolling(x=3,y=3,center=True,min_periods=1).max()
# # todo 复合条件的
# Mask=np.logical_or(np.logical_and(cond_pr_roll,arr2),np.logical_and(cond_tem_roll,arr1))
# # TODO 排除时空完全重合的
# Mask_Substract=np.logical_xor(Mask,np.logical_and(arr1,arr2))
#
# print(2)

filename=["F:\Results\CETS_ERA5\CE_Frequency_ERA5.h5",
"F:\Results\CETS_ERA5\CE_Return_ERA5.h5",
"F:\Results\CETS_ERA5\CE_Start_ERA5.h5",
"F:\Results\CETS_ERA5\CE_Amplitude_ERA5.h5",
"F:\Results\CETS_ERA5\CE_Duration_ERA5.h5",
"F:\Results\CETS_ERA5\CE_End_ERA5.h5"]
for i in filename:
    print(i)
    data=xr.open_dataset(i)
    data.to_netcdf(i.split('.')[0]+".nc")
