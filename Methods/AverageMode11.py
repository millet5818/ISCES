import numpy as np
import xarray as xr
import pandas as pd
import os

def nc_to_tiff(data, variable_name, lon_name, lat_name):
    variable_data = data
    variable_data.rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=True)
    variable_data.rio.write_crs('epsg:4326', inplace=True)  # 对数据进行投影
    output_filename = f'{variable_name}.tiff'
    output_filepath = os.path.join(output_path, output_filename)
    variable_data.rio.to_raster(output_filepath)

output_path = "E:\CE_DATA\Data_Processing\Average_Mode\\"  # 输出结果所在文件夹
# Type=['CE','CES','CET','CETS']
Type=['CES']
SSP=['Historical']
# SSP=['Historical','126','245','585']
Mode=['ACCESS-CM2','BCC-CSM2-MR','CanESM5','CMCC-ESM2','GFDL-ESM4','IPSL-CM6A-LR','MIROC6','NESM3']
filePath='F:\Results\\'
# Variable=['Frequency','Amplitude','End','Start','Duration']
Variable=['Frequency']
for ty in range(len(Type)):
    for v in range(len(Variable)):
        for s in range(len(SSP)):
            Files_name=[filePath+F'{Type[ty]}_{i}'+F"\\CE_{Variable[v]}_"+SSP[s]+".nc" for i in Mode]
            datasets=[xr.open_dataset(p) for p in Files_name]
            for i in range(len(datasets)):
                    datasets[i]['time'] = datasets[i]['time'].astype('datetime64[ns]')
                    datasets[i] = datasets[i].rename({'CE_'+Variable[v]: Mode[i]})
            combined_datasets = xr.merge(datasets)
            CE_Frequency_Map = combined_datasets.sel(quantile=0.7).mean(dim='time')
            CE_Frequency_Map_array=CE_Frequency_Map.to_array()
            results=np.mean(CE_Frequency_Map_array,axis=(0))
            variable_name = f'{Type[ty]}_{Variable[v]}_All_{SSP[s]}'  # 选择变量名称
            lon_name, lat_name = 'lon', 'lat'  # nc文件中经纬度的名称
            nc_to_tiff(results, variable_name, lon_name, lat_name)
            datasets=None
            print('处理完成！！')


