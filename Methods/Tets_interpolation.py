

import salem
import xarray as xr
import numpy as np
import geopandas as gpd

filename="F:\Experiments\Original_Data\ERA5\Max_2mTem\Maximum 2m temperature since previous post-processing_1941.nc"
outputFileName="F:\Experiments\Original_Data\ERA5\pr_day_ACCESS-CM2_ssp126_r1i1p1f1_gn_2015.nc"
dataset= xr.open_dataset(filename)
def DisplayLatLonVariables(file):
        lat_var_name = None
        lon_var_name = None
        variables = [v for v in file.variables]
        for var in variables:
            if 'lat' in var.lower():
                lat_var_name = var
            elif 'lon' in var.lower():
                lon_var_name = var
        return lat_var_name,lon_var_name
def DisplayVariables(file):
        for i in file.variables.keys():
            if len(file.variables[i].shape) > 1:
                return i
shp_file='../Data/dazhou.shp'
continent_file=gpd.read_file(shp_file)
showVariable= DisplayVariables(dataset)
u = dataset[showVariable]  # 例如u10组件
latV,lonV=DisplayLatLonVariables(dataset)
lat =dataset[latV]
lon = dataset[lonV]
target_lats = np.linspace(lat.min(), lat.max(), num=1200)  # 假设是每其中一个
target_lons = np.linspace(lon.min(), lon.max(), num=2400)  # 经度和纬度都降采样为原来的一半
u_downsampled = dataset.interp(latitude=target_lats, longitude=target_lons,method="linear")
u_downsampled=u_downsampled.astype(dataset.dtypes)
if u_downsampled.longitude.data.min() >= 0 and u_downsampled.longitude.data.max() <= 360:
    u_downsampled['longitude_Ad'] = xr.where(u_downsampled['longitude'] > 180, u_downsampled['longitude'] - 360, u_downsampled['longitude'])
    u_downsampled = (u_downsampled.swap_dims({'longitude': 'longitude_Ad'}).sel(**{'longitude_Ad': sorted(u_downsampled.longitude_Ad)}).drop('longitude'))
    u_downsampled = u_downsampled.rename({'longitude_Ad': 'longitude'})
    #u_downsampled['longitude'] = np.linspace(-180, 180, num=1800)
    u_downsampled.longitude.attrs['units'] = 'degrees_east'
u_downsampled=u_downsampled.salem.roi(shape=continent_file)
enc = {}
for k in u_downsampled.data_vars:
        if u_downsampled[k].ndim < 2:
            continue
        enc[k] = {
            "zlib": True,
            "complevel": 3,
            "fletcher32": True,
            # "chunksizes": tuple(map(lambda x: x//2, u_downsampled[k].shape))
            # "chunksizes": (1,600,1440)
        }
u_downsampled.to_netcdf(outputFileName,format='NETCDF4', engine='netcdf4',encoding=enc)


# todo 从-180到180改为0-360
# lon_name = 'lon'  # whatever name is in the data
# ds['longitude_adjusted'] = xr.where(ds[lon_name] < 0, ds[lon_name] % 360, \
#                                     ds[lon_name])
# ds = (
#     ds
#     .swap_dims({lon_name: 'longitude_adjusted'})
#     .sel(**{'longitude_adjusted': sorted(ds.longitude_adjusted)})
#     .drop(lon_name))
# ds = ds.rename({'longitude_adjusted': lon_name})