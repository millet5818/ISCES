"""
Test CE
"""
import warnings
warnings.filterwarnings("ignore")
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
import xclim.indicators
from xclim import ensembles
import xclim.indices as xi
import xclim.core.units
import dask
from xarray import open_mfdataset
from xclim import testing
from glob import glob
from xclim.core.calendar import percentile_doy,resample_doy
from xclim.indices.generic import threshold_count,compare
from xclim.ensembles import create_ensemble,ensemble_percentiles
from xclim.indices import days_over_precip_thresh
from dask.distributed import Client
from dask.diagnostics import ProgressBar

from  xclim.indicators.atmos import warm_and_wet_days



File_Pr_List="F:\Experiments\Data_Processing\ERA5_0.15_Binear\Pr\\*.nc"
File_Tem_List="F:\Experiments\Data_Processing\ERA5_0.15_Binear\Max_Tem\\*.nc"
Threshold_Pr="F:\Experiments\Data_Processing\Results\ERA5_PR_1\Threshold.nc"
Threshold_Tem="F:\Experiments\Data_Processing\Results\ERA5_Tem_1\Threshold.nc"

# TODO  CE_Duration复合事件开始到结束的持续总天数，（Duration）,称为，复合事件发生的总天数

# TODO  CE_Frequency 复合事件开始到结束的持续总发生数量，（Frequency），称为复合事件一年发生的次数

# TODO CE_Amplitude 复合事件中发生持续天数最长的天数（amplitude）称为波幅

# TODO CE_Reture 复合事件的重放周期(返回期)

# TODO CE_Ratio_PR 强降雨对复合事件的贡献率

# TODO CE_Ratio_TEM 高温对复合事件的贡献率

# TODO CE_Start 第一起复合事件的开始时间

# TODO CE_End 最后一起复合事件的结束时间

# TODO CE_IS 复合事件的强度

# TODO CE_Height 每一起复合事件的高度(即每次事件发生和结束的时间之差），这里可能多张图，因为一个格点，一年可能有多起复合事件 ,即 CE_Amplitude 属于 CE_Height 的集合

# TODO CE_Area 每一起复合事件的面积大小,平面投影面积

# TODO 窗口下的比较

# arr = pram.rolling(time=window).sum(skipna=False)
# return arr.resample(time=freq).max(dim="time").assign_attrs(units=pram.units)
# Demo
arr = xr.DataArray([[1,0,1,0,0],[1,1,1,0,0],[0,0,0,0,0],[0,0,0,1,0],[0,0,0,0,1]], dims=("x", "y"))
r = arr.rolling(y=1, center=True, min_periods=2).max()
r1 = arr.rolling(x=2,y=2, center=True, min_periods=1).max()
r2 = arr.rolling(x=2,y=2, center=True, min_periods=1).max()
mask=np.logical_and(r1,r2)
