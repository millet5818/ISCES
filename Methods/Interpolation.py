import xarray as xr
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import salem
import geopandas as gpd


class Interpolate_Way():
    def __init__(self,filename):
        self.filename=filename
        self.dataset= xr.open_dataset(filename)  # 替换为ERA5数据的实际路径
        # 选择需要的变量和经纬度坐标
        showVariable= self.DisplayVariables(self.dataset)
        self.u = self.dataset[showVariable]  # 例如u10组件
        self.latV,self.lonV=self.DisplayLatLonVariables(self.dataset)
        self.lat = self.dataset[self.latV]
        self.lon = self.dataset[self.lonV]
    def GetResolution(self):
        return self.lat.values.min(),self.lat.values.max(),self.lon.values.min(),self.lon.values.max(),len(self.lat),len(self.lon)
    def Regular_Interpolator(self,shp_file,Method,output,x_lon,y_lat,lat_min,lat_max,lon_min,lon_max):
        # 定义目标网格的经纬度范围和间隔
        outputFileName = output + "/" + self.filename.split('/')[-1]
        # target_lats = np.linspace(self.lat.min(), self.lat.max(), num=y_lat)  # 假设是每其中一个
        # target_lons = np.linspace(self.lon.min(), self.lon.max(), num=x_lon)  # 经度和纬度都降采样为原来的一半
        target_lats = np.arange(float(lat_min),float(lat_max),float(y_lat))
        target_lons = np.arange(float(lon_min),float(lon_max),float(x_lon))
        if self.latV=="latitude":
            u_downsampled=self.dataset.interp(latitude=target_lats,longitude=target_lons,method=self.Methods_Selected(Method))
        elif self.latV=="lat":
            u_downsampled = self.dataset.interp(lat=target_lats, lon=target_lons,method=self.Methods_Selected(Method))
        # 保存或进行后续分析
        u_downsampled = u_downsampled.astype(self.dataset.dtypes)
        if self.lon.min() >= 0 and self.lon.max() <= 360:
            u_downsampled['longitude_Ad'] = xr.where(u_downsampled[self.lonV] > 180, u_downsampled[self.lonV] - 360,
                                                     u_downsampled[self.lonV])
            u_downsampled = (u_downsampled.swap_dims({self.lonV: 'longitude_Ad'}).sel(
                **{'longitude_Ad': sorted(u_downsampled.longitude_Ad)}).drop(self.lonV))
            u_downsampled = u_downsampled.rename({'longitude_Ad': self.lonV})
            u_downsampled[self.lonV].attrs['units'] = 'degrees_east'

        if  len(shp_file)!=0:
            continent_file = gpd.read_file(shp_file[0])
            u_downsampled = u_downsampled.salem.roi(shape=continent_file)
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
        u_downsampled.to_netcdf(outputFileName, format='NETCDF4', engine='netcdf4', encoding=enc)

    def Statistical_Dowscaling(self):
        "统计降尺度"
        """
        Delta变化方法是一种比较简单但是很常用的降尺度方法。主要步骤是:①比较所选用的每个GCM模拟3个未来时段(2020s.2050s和2080s)不同时期的月平均降雨、
        气温数据与当前时段的月平均降雨、气温数据,观察并计算出两者之间的关系;
        ②)选择研究流域各个水文站点历史观测月平均降雨和气温数据，利用CCMs模拟得到的关系,计算出流域内各个水文站点3个未来时段的月平均降雨和气温数据。
        """
        print("统计降尺度")



    def Delta_Downscaling(self):
        print("Delta降尺度")

    def QM_Downscaling(self):
        print("Quantile Mapping")

    def PCA_Downscaling(self):
        print("Principal Components")



    def DisplayVariables(self, file):
        for i in file.variables.keys():
            if len(file.variables[i].shape) > 1:
                return i
    def DisplayLatLonVariables(self,file):
        lat_var_name = None
        lon_var_name = None
        variables = [v for v in file.variables]
        for var in variables:
            if 'lat' in var.lower():
                lat_var_name = var
            elif 'lon' in var.lower():
                lon_var_name = var
        return lat_var_name,lon_var_name

    def Methods_Selected(self,Method):
        if Method=="Nearest":
            return "nearest"
        elif Method=="Cubic":
            return "cubic"
        elif Method=="Bilinear":
            return "linear"
