import xarray as xr

import xarray as xr
import os
import pandas as pd


def nc_to_tiff(data, variable_name, lon_name, lat_name):
    variable_data = data[variable_name].sel(quantile=0.7)
    variable_data.rio.set_spatial_dims(lon_name, lat_name, inplace=True)
    variable_data.rio.write_crs('epsg:4326', inplace=True)  # 对数据进行投影
    variable_data=variable_data.rename({lon_name: "x", lat_name: "y"})
    time_list = variable_data['time'].values
    for timei in time_list:
        # 根据时间筛选数据
        variable_time_data = variable_data.sel(time=timei)
        variable_time_data = variable_data.sel(time=timei)
        # 获取时间字符串
        timei = pd.to_datetime(str(timei))
        timestr = timei.strftime('%Y%m%d')
        # 转出文件的名称为：变量名+时间
        output_filename = f'{variable_name}_{timestr}.tiff'
        output_filepath = os.path.join(output_path, output_filename)
        variable_time_data.rio.to_raster(output_filepath)


def nc_to_tiff_User(data, variable_name, lon_name, lat_name):
    variable_data = data[variable_name].sel(quantile=0.8)
    variable_data.rio.set_spatial_dims(lon_name, lat_name, inplace=True)
    variable_data.rio.write_crs('epsg:4326', inplace=True)  # 对数据进行投影
    variable_data=variable_data.rename({lon_name: "x", lat_name: "y"})
    time_list = variable_data['time'].values
    # variable_time_data = variable_data.sel(time=slice("2015","2023")).mean(dim='time')
    variable_time_data = variable_data.sel(time=slice("2015","2100")).sum(dim='time')
    # 转出文件的名称为：变量名+时间
    output_filename = f'{variable_name}_2100_585.tiff'
    output_filepath = os.path.join(output_path, output_filename)
    variable_time_data.rio.to_raster(output_filepath)


input_path ="E:\CE_DATA\Data_Processing\Average_Mode\CE\CMIP6\CE_Frequency_585.nc" # nc文件所在文件夹
output_path = "D:\zhaozheng\projects\Global Risk\累积因子\\"  # 输出结果所在文件夹
data = xr.open_dataset(input_path)  # nc文件数据
variable_name = 'CE_Frequency'  # 选择变量名称
lon_name, lat_name = 'lon', 'lat'  # nc文件中经纬度的名称
nc_to_tiff_User(data, variable_name, lon_name, lat_name)
data.close()  # 关闭数据集
data = None
print('处理完成！！')
