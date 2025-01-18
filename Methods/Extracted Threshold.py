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
from xclim.core.units import amount2rate
import xclim
from dask.diagnostics import ProgressBar

from scipy.signal import convolve,choose_conv_method,convolve2d
from xclim.indices import run_length as rl
from  xclim.core.units import to_agg_units
import xarray
from functools import partial
FileNameList_Tem="F:\Datasets\CMIP6\CMIP6_0.15_Binear\Max_Tem\IPSL-CM6A-LR\\Historical\\*.nc"
Time_Base=np.arange(1950,2015,1)
Files_Tem = open_mfdataset(FileNameList_Tem,parallel=True,concat_dim="time",chunks={'time': -1,'lat':200, 'lon': 200}, combine="nested",data_vars='minimal', coords='minimal', compat='override')
Files_Tem['tasmax'] = xclim.core.units.convert_units_to(Files_Tem['tasmax'], "degC")
Files_Tem_Base=Files_Tem.where(Files_Tem.time.dt.year.isin(Time_Base),drop=True)
Tem_theshold = Files_Tem_Base['tasmax'].chunk({"time": -1, "lat": 200, "lon": 200}).quantile([0.7,0.8,0.9], dim="time", keep_attrs=True)
aaa=Tem_theshold.rename('Threshold').to_netcdf("E:\CE_DATA\Data_Processing\Process_Results\Threshold\Tem\IPSL-CM6A-LR\Historical_Threshold_Tem.nc",compute=False)
with ProgressBar():
    results = aaa.compute()

