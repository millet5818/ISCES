import warnings
warnings.filterwarnings("ignore")
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
import xclim.indicators
#from distributed import Client
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
# pip install dask[complete] distributed --upgrade
# pip install xarray[complete]
from dask.diagnostics import ProgressBar


# TODO Start time 15.54

if __name__ == '__main__':
    # client=Client(memory_limit='20GB',n_workers=1)
    # client=Client(processes=False,n_workers=1,threads_per_worker=10, memory_limit='15GB')# TODO 不出现任何消息输出的方式 or 把下面的代码放在一起
    client=Client(n_workers=1,threads_per_worker=20, memory_limit='10GB')

# client=Client(processes=False)
outputfile='E:\CE_DATA\Data_Processing\Process_Results\ssss4.nc'
FileNameList=[
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1941.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1942.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1943.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1944.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1945.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1946.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1947.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1948.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1949.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1950.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1951.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1952.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1953.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1954.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1955.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1956.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1957.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1958.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1959.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1960.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1961.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1962.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1963.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1964.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1965.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1966.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1967.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1968.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1969.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1970.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1971.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1972.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1973.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1974.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1975.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1976.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1977.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1978.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1979.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1980.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1981.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1982.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1983.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1984.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1985.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1986.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1987.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1988.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1989.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1990.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1991.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1992.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1993.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1994.nc",
"E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1995.nc"]
# FileNameList=[
# "E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1941.nc",
# "E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1942.nc",
# "E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1943.nc",
# "E:\CE_DATA\Data_Processing\ERA5\Pr\\total_precipitation_1944.nc"]
# Files=open_mfdataset(FileNameList,concat_dim="time", combine="nested",data_vars='minimal', coords='minimal', compat='override', chunks={'time': -1,'latitude': 100, 'longitude': 100})
# Files=open_mfdataset(FileNameList,parallel=True,concat_dim="time",chunks={'time': -1,'latitude': 100, 'longitude': 100}, combine="nested",data_vars='minimal', coords='minimal', compat='override')
Files=open_mfdataset(FileNameList,parallel=True,concat_dim="time", chunks={'time': -1,'latitude': 100, 'longitude': 100},combine="nested")
Files['tp'] = xclim.core.units.amount2rate(Files['tp'], out_units="mm/d")
# todo 两者都可以
p75 = Files.tp.chunk({"time": len(Files.time), "latitude": 100, "longitude": 100}).quantile(0.75, dim="time", keep_attrs=True)
r75p = days_over_precip_thresh(Files.tp, p75)
delayed_obj =r75p.load().to_netcdf(outputfile, format='NETCDF4', engine='netcdf4',mode='w',compute=False)
# delayed_obj =r75p.chunk({"time": len(Files.time), "latitude": 100, "longitude": 100}).to_netcdf(outputfile, format='NETCDF4', engine='netcdf4',mode='w',compute=False)
# delayed_obj =r75p.load().to_netcdf(outputfile, format='NETCDF4', engine='netcdf4',mode='w',compute=False)
with ProgressBar():
    results = delayed_obj.compute()
