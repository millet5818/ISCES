import os
import datetime
from datetime import date, timedelta
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import warnings
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
from netCDF4 import Dataset as netcdf_dataset
import tkinter
from tkinter import *
from tkinter import StringVar
from tkinter import ttk
from tkinter.filedialog import askdirectory
import cmaps
import rioxarray
from scipy.stats import linregress
from dask.array import apply_along_axis
import threading
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Arial'

# 将未来气候数据分割为逐年数据
def Get_Yearly_Data(inPath_NC_1, outPath_NC_1):
	# 将所有数据合并
	dataset_Tmax_list_1 = []
	model_1 = ''
	for dirPath, dirname, filenames in os.walk(inPath_NC_1):
		for filename in filenames:
			inPath_NC_2 = os.path.join(inPath_NC_1, filename)
			dataset_Tmax_1 = xr.open_dataset(inPath_NC_2)
			dataset_Tmax_list_1.append(dataset_Tmax_1)	
			model_1 = filename.split('tasmax_day_')[1].split('_r1i1p1f2')[0]
	print(model_1)
	dataset_Tmax_Year_1 = dataset_Tmax_list_1[0]
	for dataset_Tmax_1 in dataset_Tmax_list_1[1: ]:
		# dataset_Tmax_Year_1 = xr.merge([dataset_Tmax_Year_1, dataset_Tmax_1])
		dataset_Tmax_Year_1 = xr.concat([dataset_Tmax_Year_1, dataset_Tmax_1], dim = 'time')
		print('Concating...')
	# dataset_Tmax_Year_1.to_netcdf(os.path.join(outPath_NC_1, 'Tmax_daily_20210101-20991231' + '.nc'))
	# 将数据按年份分割
	for year in range(2021, 2101):
		date_Start_1 = str(year) + '-01-01'   # start date
		date_End_1 = str(year) + '-12-31'   # end date
		dataset_Tmax_2 = dataset_Tmax_Year_1.sel(time = slice(date_Start_1, date_End_1))
		# 转换日期格式
		# time_Datetime_1 = dataset_Tmax_2.indexes['time'].to_datetimeindex()
		time_Datetime_1 = dataset_Tmax_2.indexes['time']
		dataset_Tmax_2['time'] = time_Datetime_1
		# 将复杂数据格式另存为简单数据格式
		tmax_1 = dataset_Tmax_2.tasmax.values - 273.15
		lon_1 = dataset_Tmax_2.lon.values
		lat_1 = dataset_Tmax_2.lat.values
		time_1 = dataset_Tmax_2.time.values
		dataset_Tmax_2 = xr.Dataset({'CAT': (['time', 'lat', 'lon'], tmax_1),}, coords = {'lon': (['lon'], lon_1), 'lat': (['lat'], lat_1), 'time': time_1})
		outPath_NC_2 = os.path.join(outPath_NC_1, model_1, '1_Temp_Daily')
		Create_Dir(outPath_NC_2)
		outPath_NC_3 = os.path.join(outPath_NC_2, model_1.lower() + '_air_temp_daily_tmax_' + str(year) + '.nc')
		gap_1 = 1.40625 # 1.25 / 1.5
		dataset_Tmax_2 = dataset_Tmax_2.roll(lon = int(180 / gap_1))
		lat_2 = np.arange(-90, 90, gap_1)
		lon_2 = np.arange(-180, 180, gap_1)
		dataset_Tmax_2['lat'] = lat_2
		dataset_Tmax_2['lon'] = lon_2
		dataset_Tmax_2.to_netcdf(outPath_NC_3)
		print(str(year) + ' is done!')

# 创建文件夹
def Create_Dir(path_1):
	if os.path.exists(path_1):
		pass
	else:
		os.makedirs(path_1)

# 数据标准化：截取、滚动和重采样
def Standardize(inPath_AT_1, gap_1, outPath_AT_1):
	Create_Dir(outPath_AT_1)
	for dirPath, dirname, filenames in os.walk(inPath_AT_1):
		for filename in filenames:
			inPath_AT_2 = os.path.join(inPath_AT_1, filename)
			dataset_AT_1 = xr.open_dataset(inPath_AT_2)
			# **************重采样***********
			dataset_AT_1 = dataset_AT_1.coarsen(lon = 4.5, boundary = 'trim').mean().coarsen(lat = 4.5, boundary = 'trim').mean()
			# **************截取陆地区域*****
			# dataset_Mask_1 = xr.open_dataset(r'J:\\2_personal\1_papers\6_heat wave variation\2_data\Base\ERA5_mask_3.nc')
			# dataset_AT_1 = xr.where(dataset_Mask_1 == 1, dataset_AT_1, np.nan)
			# **************滚动************
			# dataset_AT_1 = dataset_AT_1.roll(lon = int(180 / gap_1))
			lat_2 = np.arange(-90, 90, gap_1)
			lon_2 = np.arange(-180, 180, gap_1)
			dataset_AT_1['lat'] = lat_2
			dataset_AT_1['lon'] = lon_2
			outPath_AT_2 = os.path.join(outPath_AT_1, filename)
			dataset_AT_1.to_netcdf(outPath_AT_2)
			print(filename + ' is done!')

# 重采样
def Interpolate(inPath_ref_1, inPath_old_1, outPath_new_1):
	dataset_1 = xr.open_dataset(inPath_ref_1)
	lat_1 = dataset_1['lat'].values
	lon_1 = dataset_1['lon'].values
	dataset_2 = xr.open_dataset(inPath_old_1)
	dataset_2 = dataset_2.interp(lat = lat_1) # specify calculation
	dataset_2 = dataset_2.interp(lon = lon_1) # specify calculation
	dataset_2.to_netcdf(outPath_new_1) # save direct to file

# 创建多线程
def MyThread(func, *args):
	thread_1 = threading.Thread(target = func, args=args) 
	thread_1.setDaemon(True) 
	thread_1.start()

# 计算CRUJRA逐日体感温度/温度
def Cal_CRUJRA_AT(inPath_NC_1, start_1, end_1, statistic_1, result_tag_1, outPath_AT_1, text_1):
	for year in range(int(start_1), int(end_1) + 1):
		outPath_AT_2 = os.path.join(outPath_AT_1, 'CRU-JRA', result_tag_1, statistic_1, '1_Temp_Daily') 
		Create_Dir(outPath_AT_2)
		outPath_AT_3 = os.path.join(outPath_AT_2, 'cru-jra_' + result_tag_1 + '_daily_' + statistic_1 + '_' + str(year) + '.nc')
		if os.path.exists(outPath_AT_3):
			pass
		else:
			variables_1 = ['tmax', 'ugrd', 'vgrd', 'pres', 'spfh']
			dataset_NC_list_1 = []
			for variable_1 in variables_1:
				inPath_NC_2 = os.path.join(inPath_NC_1, variable_1, 'crujra.v2.2.5d.' + variable_1 + '.' + str(year) + '.365d.noc.nc')
				dataset_NC_1 = xr.open_dataset(inPath_NC_2)
				dataset_NC_list_1.append(dataset_NC_1.rename({variable_1: 'CAT'}))
			dataset_Year_List_1 = []
			date_Start_1 = date(year, 1, 1)   # start date
			date_End_1 = date(year, 12, 31)   # end date
			date_List = pd.date_range(date_Start_1, date_End_1, freq = 'd')
			for date_1 in date_List:
				try:
					dataset_Day_List_1 = []
					hour_List = pd.date_range(date_1, date_1 + timedelta(hours = 18), freq = '6h')
					for hour_1 in hour_List:
						dataset_hour_list_1 = []
						for dataset_NC_2 in dataset_NC_list_1:
							dataset_hour_list_1.append(dataset_NC_2.sel(time = hour_1.strftime('%Y-%m-%d %H:%M:%S')))
						if result_tag_1 == 'app_temp':
							# 计算体感温度
							# 计算相对湿度
							dataset_RH_Hour_1 = (dataset_hour_list_1[4] * (dataset_hour_list_1[3] / 100) / (0.378 * dataset_hour_list_1[4] + 0.622)) / \
									(6.112 * pow(np.e, (17.67 * (dataset_hour_list_1[0] - 273.15)) / ((dataset_hour_list_1[0] - 273.15) + 243.5)))
							# 计算水汽压
							dataset_WP_Hour_1 = dataset_RH_Hour_1 / 100 * 6.105 * pow(np.e, 17.27 * (dataset_hour_list_1[0] - 273.15) / (237.7 + (dataset_hour_list_1[0] - 273.15)))
							# 计算风速
							dataset_WS_Hour_1 = pow(dataset_hour_list_1[1] * dataset_hour_list_1[1] + dataset_hour_list_1[2] * dataset_hour_list_1[2], 0.5)
								# 计算体感温度
							dataset_AT_Hour_1 = (dataset_hour_list_1[0] - 273.15) + 0.33 * dataset_WP_Hour_1 - 0.7 * dataset_WS_Hour_1 - 4
							dataset_Day_List_1.append(dataset_AT_Hour_1)	
						elif result_tag_1 == 'air_temp':
							# 计算温度
							dataset_AT_Hour_1 = dataset_hour_list_1[0] - 273.15
							dataset_Day_List_1.append(dataset_AT_Hour_1)
					# 将逐3小时体感温度合并到逐日
					dataset_Day_1 = xr.concat(dataset_Day_List_1, dim = 'time')
					# 计算逐日最高', '平均和最低体感温度
					if statistic_1 == 'tmax':
						dataset_Day_2 = dataset_Day_1.max(dim = 'time')
					elif statistic_1 == 'tmean':
						dataset_Day_2 = dataset_Day_1.mean(dim = 'time')
					else:
						dataset_Day_2 = dataset_Day_1.min(dim = 'time')
					# 为逐日体感温度写入当日日期
					time_Day_1 = pd.to_datetime([str(date_1.year) + str(date_1.month).zfill(2) + str(date_1.day).zfill(2)])
					time_Day_2 = xr.DataArray(time_Day_1, [('time', time_Day_1.dayofyear)])
					dataset_Day_3 = dataset_Day_2.expand_dims(time = time_Day_2)
					# 逐日体感温度列表
					dataset_Year_List_1.append(dataset_Day_3)
					print(date_1.strftime('%Y-%m-%d') + ' is done.')
					text_1.insert(INSERT, date_1.strftime('%Y-%m-%d') + ' is done.\n')
					text_1.see(END)
					text_1.update_idletasks()
				except Exception as e:
					print(str(date_1) + 'does not exist.')
			# 将逐日体感温度合并到年
			doy_1 = 1
			dataset_Year_1 = dataset_Year_List_1[0]
			for dataset_Day_1 in dataset_Year_List_1[1: ]:
				dataset_Year_1 = xr.concat([dataset_Year_1, dataset_Day_1], dim = 'time')
				print('Concating the ' + str(doy_1) + ' day!')
				doy_1 = doy_1 + 1
			# plot_Day_1 = dataset_Year_1.CAT.isel(time = 2)
			# plot_Day_1.plot()
			# plt.show()
			dataset_Year_1.to_netcdf(outPath_AT_3)
			print('Finished!')

# 计算ERA5逐日体感温度/温度
def Cal_ERA5_AT(inPath_NC_1, start_1, end_1, statistic_1, result_tag_1, outPath_AT_1, text_1):
	for year in range(int(start_1), int(end_1) + 1):
		outPath_AT_2 = os.path.join(outPath_AT_1, 'ERA5', result_tag_1, statistic_1, '1_Temp_Daily') 
		Create_Dir(outPath_AT_2)
		outPath_AT_3 = os.path.join(outPath_AT_2, 'era5_' + result_tag_1 + '_daily_' + statistic_1 + '_' + str(year) + '.nc')
		if os.path.exists(outPath_AT_3):
			pass
		else:
			# variables_1 = ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature', '2m_temperature']
			# variables_2 = ['u10', 'v10', 'd2m', 't2m']
			variables_1 = ['2m_temperature']
			variables_2 = ['t2m']
			index_1 = 0
			dataset_NC_list_1 = []
			for variable_1 in variables_1:
				inPath_NC_2 = os.path.join(inPath_NC_1, variable_1 + '_' + str(year) + '.nc')
				dataset_NC_1 = xr.open_dataset(inPath_NC_2)
				dataset_NC_list_1.append(dataset_NC_1.rename({variables_2[index_1]: 'CAT'}))
				index_1 = index_1 + 1
			dataset_Year_List_1 = []
			date_Start_1 = date(year, 1, 1)   # start date
			date_End_1 = date(year, 12, 31)   # end date
			date_List = pd.date_range(date_Start_1, date_End_1, freq = 'd')
			for date_1 in date_List:
				dataset_Day_List_1 = []
				hour_List = pd.date_range(date_1, date_1 + timedelta(hours = 21), freq = '3h')
				for hour_1 in hour_List:
					dataset_hour_list_1 = []
					for dataset_NC_2 in dataset_NC_list_1:
						dataset_hour_list_1.append(dataset_NC_2.sel(time = hour_1.strftime('%Y-%m-%d %H:%M:%S')))
					if result_tag_1 == 'app_temp':
						# 计算体感温度
						# 计算相对湿度
						dataset_RH_Hour_1 = (dataset_hour_list_1[2] + 19.2 - 0.84 * dataset_hour_list_1[3]) / (0.1980 + 0.0017 * dataset_hour_list_1[3])
						# 计算水汽压
						dataset_WP_Hour_1 = dataset_RH_Hour_1 / 100 * 6.105 * pow(np.e, 17.27 * (dataset_hour_list_1[3] - 273.15) / (237.7 + (dataset_hour_list_1[3] - 273.15)))
						# 计算风速
						dataset_WS_Hour_1 = pow(dataset_hour_list_1[0] * dataset_hour_list_1[0] + dataset_hour_list_1[1] * dataset_hour_list_1[1], 0.5)
						# 计算体感温度
						dataset_AT_Hour_1 = (dataset_hour_list_1[3] - 273.15) + 0.33 * dataset_WP_Hour_1 - 0.7 * dataset_WS_Hour_1 - 4
						dataset_Day_List_1.append(dataset_AT_Hour_1)	
					elif result_tag_1 == 'air_temp':
						# 计算温度
						dataset_AT_Hour_1 = dataset_hour_list_1[0] - 273.15
						dataset_Day_List_1.append(dataset_AT_Hour_1)
				# 将逐3小时体感温度合并到逐日
				dataset_Day_1 = xr.concat(dataset_Day_List_1, dim = 'time')
				# 计算逐日最高', '平均和最低体感温度
				if statistic_1 == 'tmax':
					dataset_Day_2 = dataset_Day_1.max(dim = 'time')
				elif statistic_1 == 'tmean':
					dataset_Day_2 = dataset_Day_1.mean(dim = 'time')
				else:
					dataset_Day_2 = dataset_Day_1.min(dim = 'time')
				# 为逐日体感温度写入当日日期
				time_Day_1 = pd.to_datetime([str(date_1.year) + str(date_1.month).zfill(2) + str(date_1.day).zfill(2)])
				time_Day_2 = xr.DataArray(time_Day_1, [('time', time_Day_1.dayofyear)])
				dataset_Day_3 = dataset_Day_2.expand_dims(time = time_Day_2)
				# 逐日体感温度列表
				dataset_Year_List_1.append(dataset_Day_3)
				print(date_1.strftime('%Y-%m-%d') + ' is done.')
				text_1.insert(INSERT, date_1.strftime('%Y-%m-%d') + ' is done.\n')
				text_1.see(END)
				text_1.update_idletasks()
			# 将逐日体感温度合并到年
			doy_1 = 1
			dataset_Year_1 = dataset_Year_List_1[0]
			for dataset_Day_1 in dataset_Year_List_1[1: ]:
				dataset_Year_1 = xr.concat([dataset_Year_1, dataset_Day_1], dim = 'time')
				print('Concating the ' + str(doy_1) + ' day!')
				doy_1 = doy_1 + 1
			# plot_Day_1 = dataset_Year_1.CAT.isel(time = 2)
			# plot_Day_1.plot()
			# plt.show()
			dataset_Year_1 = dataset_Year_1.rename({'latitude': 'lat'})
			dataset_Year_1 = dataset_Year_1.rename({'longitude': 'lon'})
			dataset_Year_1.to_netcdf(outPath_AT_3)
			print('Finished!')

# 计算GLDAS逐日体感温度/温度
def Cal_GLDAS_AT(inPath_NC_1, start_1, end_1, statistic_1, result_tag_1, outPath_AT_1, text_1):
	for year in range(int(start_1), int(end_1) + 1):
		outPath_AT_2 = os.path.join(outPath_AT_1, 'GLDAS', result_tag_1, statistic_1, '1_Temp_Daily') 
		Create_Dir(outPath_AT_2)
		outPath_AT_3 = os.path.join(outPath_AT_2, 'gldas_' + result_tag_1 + '_daily_' + statistic_1 + '_' + str(year) + '.nc')
		if os.path.exists(outPath_AT_3):
			pass
		else:
			dataset_Year_List_1 = []
			date_Start_1 = date(year, 1, 1)   # start date
			date_End_1 = date(year, 12, 31)   # end date
			date_List = pd.date_range(date_Start_1, date_End_1, freq = 'd')
			for date_1 in date_List:
				dataset_Day_List_1 = []
				hour_List = pd.date_range(date_1, date_1 + timedelta(hours = 21), freq = '3h')
				for hour_1 in hour_List:
					filename_Hour_1 = 'GLDAS_NOAH025_3H.A' + hour_1.strftime('%Y%m%d.%H') + '00.020.nc4.SUB.nc4'
					if year > 2000:
						filename_Hour_1 = 'GLDAS_NOAH025_3H.A' + hour_1.strftime('%Y%m%d.%H') + '00.021.nc4.SUB.nc4'
					inPath_Hour_1 = os.path.join(inPath_NC_1, filename_Hour_1)
					dataset_NC_1 = xr.open_dataset(inPath_Hour_1)
					if result_tag_1 == 'app_temp':
						# 计算逐3小时体感温度					
						# 计算相对湿度
						dataset_NC_2 = dataset_NC_1.assign(Crh = (dataset_NC_1['Qair_f_inst'] * (dataset_NC_1['Psurf_f_inst'] / 100) / (0.378 * dataset_NC_1['Qair_f_inst'] + 0.622)) / \
							(6.112 * pow(np.e, (17.67 * (dataset_NC_1['Tair_f_inst'] - 273.15)) / ((dataset_NC_1['Tair_f_inst'] - 273.15) + 243.5))))
						# 计算水汽压
						dataset_NC_3 = dataset_NC_2.assign(Cwp = dataset_NC_2['Crh'] / 100 * 6.105 * pow(np.e, 17.27 * (dataset_NC_2['Tair_f_inst'] - 273.15) / (237.7 + (dataset_NC_2['Tair_f_inst'] - 273.15))))
						# 计算体感温度
						dataset_NC_4 = dataset_NC_3.assign(CAT = (dataset_NC_3['Tair_f_inst'] - 273.15) + 0.33 * dataset_NC_3['Cwp'] - 0.7 * dataset_NC_3['Wind_f_inst'] - 4)
						dataset_AT_1 = dataset_NC_4[['CAT']]
					elif result_tag_1 == 'air_temp':
						# 计算逐3小时气温
						dataset_NC_2 = dataset_NC_1.assign(CAT = (dataset_NC_1['Tair_f_inst'] - 273.15))
						dataset_AT_1 = dataset_NC_2[['CAT']]						
					dataset_Day_List_1.append(dataset_AT_1)
				# 将逐3小时体感温度合并到逐日
				dataset_Day_1 = xr.concat(dataset_Day_List_1, dim = 'time')
				# outPath_AT_3 = os.path.join(outPath_AT_1, 'AT_Day_' + str(year) + str(month).zfill(2) + str(day).zfill(2) + '.nc') 
				# dataset_Day_1.to_netcdf(outPath_AT_3)
				# 计算逐日最高', '平均和最低体感温度
				if statistic_1 == 'tmax':
					dataset_Day_2 = dataset_Day_1.max(dim = 'time')
				elif statistic_1 == 'tmean':
					dataset_Day_2 = dataset_Day_1.mean(dim = 'time')
				else:
					dataset_Day_2 = dataset_Day_1.min(dim = 'time')
				# 为逐日体感温度写入当日日期
				time_Day_1 = pd.to_datetime([str(date_1.year) + str(date_1.month).zfill(2) + str(date_1.day).zfill(2)])
				time_Day_2 = xr.DataArray(time_Day_1, [('time', time_Day_1.dayofyear)])
				dataset_Day_3 = dataset_Day_2.expand_dims(time = time_Day_2)
				# 逐日体感温度列表
				dataset_Year_List_1.append(dataset_Day_3)
				print(date_1.strftime('%Y-%m-%d') + ' is done.')
				text_1.insert(INSERT, date_1.strftime('%Y-%m-%d') + ' is done.\n')
				text_1.see(END)
				text_1.update_idletasks()
			# 将逐日体感温度合并到年
			doy_1 = 1
			dataset_Year_1 = dataset_Year_List_1[0]
			for dataset_Day_1 in dataset_Year_List_1[1: ]:
				dataset_Year_1 = xr.concat([dataset_Year_1, dataset_Day_1], dim = 'time')
				print('Concating the ' + str(doy_1) + ' day!')
				doy_1 = doy_1 + 1
			# plot_Day_1 = dataset_Year_1.CAT.isel(time = 2)
			# plot_Day_1.plot()
			# plt.show()
			# print(dataset_Year_1)
			# dataset_Year_1 = dataset_Year_1.coarsen(lon = 2, boundary = 'trim').mean().coarsen(lat = 2, boundary = 'trim').mean()
			dataset_Year_1.to_netcdf(outPath_AT_3)
			print('Finished!')

# 计算逐日高温阈值
def Cal_Ref_1(inPath_AT_1, strat_1, end_1, range_1, percentile_1, text_1):
	outPath_Ref_1 = os.path.join(os.path.dirname(inPath_AT_1), '2_Temp_Ref')
	Create_Dir(outPath_Ref_1)
	filename_Demo_1 = os.listdir(inPath_AT_1)[0].rsplit('_', 1)[0]
	filename_Ref_1 = filename_Demo_1 + '_' + str(percentile_1) + '_' + str(range_1) + '_' + str(strat_1) + '-' + str(end_1) + '.nc'
	outPath_Ref_2 = os.path.join(outPath_Ref_1, filename_Ref_1)
	if os.path.exists(outPath_Ref_2):
		pass
	else:
		# 高温阈值数据列表
		dataset_Ref_list_1 = []
		doy_List_1 = []
		# 计算每一个日期的高温阈值
		for month in range(1, 13):
			for day in range(1, 32):
				# 计算该日高温阈值所用到的体感温度
				dataset_AT_list_1 = []
				# 计算该日高温阈值所用到的年份
				for year in range(int(strat_1) + 1, int(end_1)):
					filename_AT_1 = filename_Demo_1 + '_' + str(year) + '.nc'
					inPath_AT_2 = os.path.join(inPath_AT_1, filename_AT_1)
					# 该日日期
					date_str_1 = str(year) + '-' + str(month).zfill(2) + '-' + str(day).zfill(2)
					try:
						date_time_1 = datetime.datetime.strptime(date_str_1, '%Y-%m-%d')
						dataset_AT_1 = xr.open_dataset(inPath_AT_2)
						# 该日前n天的日期
						date_time_2 = date_time_1 - datetime.timedelta(days = int(range_1)) 
						# 该日后n天的日期
						date_time_3 = date_time_1 + datetime.timedelta(days = int(range_1))
						# 不跨年份的时间窗口
						dataset_AT_5 = dataset_AT_1.sel(time = slice(date_time_2, date_time_3))
						# 跨年份的时间窗口
						if date_time_2.year == year - 1:
							filename_AT_2 = filename_Demo_1 + '_' + str(year - 1) + '.nc'
							inPath_AT_3 = os.path.join(inPath_AT_1, filename_AT_2)
							dataset_AT_2 = xr.open_dataset(inPath_AT_3)
							dataset_AT_3 = dataset_AT_2.sel(time = slice(date_time_2, str(year - 1) + '-12-31'))
							dataset_AT_4 = dataset_AT_1.sel(time = slice(str(year) + '-01-01', date_time_3))
							dataset_AT_5 = xr.concat([dataset_AT_3, dataset_AT_4], dim = 'time')
						elif date_time_3.year == year + 1:
							filename_AT_2 = filename_Demo_1 + '_' + str(year + 1) + '.nc'
							inPath_AT_3 = os.path.join(inPath_AT_1, filename_AT_2)
							dataset_AT_2 = xr.open_dataset(inPath_AT_3)
							dataset_AT_3 = dataset_AT_1.sel(time = slice(date_time_2, str(year) + '-12-31'))
							dataset_AT_4 = dataset_AT_2.sel(time = slice(str(year + 1) + '-01-01', date_time_3))
							dataset_AT_5 = xr.concat([dataset_AT_3, dataset_AT_4], dim = 'time')		
						dataset_AT_list_1.append(dataset_AT_5)
						print(date_str_1 + ' is done!')
						text_1.insert(INSERT, date_str_1 + ' is done!\n')
						text_1.see(END)
						text_1.update_idletasks()
					except Exception as e:
						print(date_str_1 + ' does not exist.')
				if len(dataset_AT_list_1) > 0:
					dataset_AT_6 = xr.concat(dataset_AT_list_1, dim = 'time')
					# 计算百分位数
					dataset_Ref_1 = dataset_AT_6.quantile(int(percentile_1) / 100, dim = 'time')
					# 写入时间维
					time_Ref_1 = pd.to_datetime([str(2000) + str(month).zfill(2) + str(day).zfill(2)])
					time_Ref_2 = xr.DataArray(time_Ref_1, [('time', time_Ref_1)])
					dataset_Ref_2 = dataset_Ref_1.expand_dims(time = time_Ref_2)
					doy_List_1.append(int(time_Ref_1.strftime('%j')[0]))
					# 追加到高温阈值数据列表
					dataset_Ref_list_1.append(dataset_Ref_2)
					print(str(month).zfill(2) + str(day).zfill(2) + ' is done.')
				else:
					print(str(month).zfill(2) + str(day).zfill(2) + ' does not exist.')
		dataset_Ref_Year_1 = dataset_Ref_list_1[0]
		doy_1 = 1
		for dataset_Ref_1 in dataset_Ref_list_1[1: ]:
			dataset_Ref_Year_1 = xr.concat([dataset_Ref_Year_1, dataset_Ref_1], dim = 'time')
			print('Concating the ' + str(doy_1) + ' day!')
			doy_1 = doy_1 + 1
		dataset_Ref_Year_1['time'] = doy_List_1
		dataset_Ref_Year_1.to_netcdf(outPath_Ref_2)
		print('Finished!')

# 计算逐年高温热浪（仅计算热浪）
def Cal_HW_1(inPath_AT_1, filename_Ref_1, con_Thre_1, dura_Thre_1, text_1):
	filename_Demo_1 = os.listdir(inPath_AT_1)[0].rsplit('_', 1)[0]
	outPath_HW_1 = os.path.join(os.path.dirname(inPath_AT_1), '4_HWE_Annual', '1_HWE')
	outPath_HW_2 = os.path.join(os.path.dirname(inPath_AT_1), '4_HWE_Annual', '2_HWDR')
	Create_Dir(outPath_HW_1)
	Create_Dir(outPath_HW_2)
	# 参考体感温度
	inPath_Ref_1 = os.path.join(os.path.dirname(inPath_AT_1), '2_Temp_Ref', filename_Ref_1)
	dataset_Ref_1 = xr.open_dataset(inPath_Ref_1)
	# 遍历每年的逐日体感温度
	for dirPath, dirname, filenames in os.walk(inPath_AT_1):
		for filename in filenames:
			year_1 = filename.rsplit('_', 1)[1].split('.nc', 1)[0]
			outPath_HW_3 = os.path.join(outPath_HW_1, filename_Demo_1 + '_' + str(con_Thre_1) + '_' + str(dura_Thre_1) + '_HWE_' + year_1 + '.nc')
			outPath_HW_4 = os.path.join(outPath_HW_2, filename_Demo_1 + '_' + str(con_Thre_1) + '_' + str(dura_Thre_1) + '_HWDR_' + year_1 + '.nc')
			if os.path.exists(outPath_HW_3):
				pass
			else:
				dataset_Tag_1 = 0
				dataset_Tag_2 = 0
				dataset_AT_Sum_1 = 0
				dataset_AT_Max_1 = 0
				dataset_Freq_1 = 0
				dataset_Dura_1 = 0
				dataset_Dmax_1 = 0
				dataset_Tavg_1 = 0
				dataset_Tmax_1 = 0
				dataset_Start_1 = 0
				dataset_End_1 = 0
				HWDR_List_1 = []
				# 读取逐日体感温度
				inPath_AT_2 = os.path.join(inPath_AT_1, filename)
				dataset_AT_1 = xr.open_dataset(inPath_AT_2)
				# 读取时间维
				time_AT_Index = dataset_AT_1.indexes['time']
				for time_AT_1 in time_AT_Index:
					# 该日体感温度
					dataset_AT_Day_1 = dataset_AT_1.sel(time = time_AT_1)
					doy_1 = int(time_AT_1.strftime('%j'))
					# 该日参考体感温度
					dataset_Ref_Day_1 = dataset_Ref_1.sel(time = doy_1)
					# 判断高温热浪
					dataset_Bool_1 = xr.where((dataset_AT_Day_1 >= con_Thre_1) & (dataset_AT_Day_1 >= dataset_Ref_Day_1), 1, 0)
					dataset_Tag_1 = xr.where(dataset_Bool_1, dataset_Tag_1 + 1, dataset_Tag_1)
					dataset_AT_Sum_1 = xr.where(dataset_Bool_1, dataset_AT_Sum_1 + dataset_AT_Day_1, dataset_AT_Sum_1)
					dataset_AT_Sum_1 = xr.where((1 - dataset_Bool_1) & (dataset_Tag_1 < dura_Thre_1), 0, dataset_AT_Sum_1)
					dataset_AT_Max_1 = xr.where(dataset_Bool_1, dataset_AT_Day_1, dataset_AT_Max_1)
					dataset_AT_Max_1 = xr.where((1 - dataset_Bool_1) & (dataset_Tag_1 < dura_Thre_1), 0, dataset_AT_Max_1)
					dataset_Tag_1 = xr.where((1 - dataset_Bool_1) & (dataset_Tag_1 < dura_Thre_1), 0, dataset_Tag_1)
					dataset_Tag_2 = xr.where(((1 - dataset_Bool_1) | (time_AT_1 == time_AT_Index[-1])) & (dataset_Tag_1 >= dura_Thre_1), dataset_Tag_1, np.nan)
					HWDR_List_1.append(dataset_Tag_2)
					dataset_Freq_1 = xr.where(((1 - dataset_Bool_1) | (time_AT_1 == time_AT_Index[-1])) & (dataset_Tag_1 >= dura_Thre_1), dataset_Freq_1 + 1, dataset_Freq_1)
					dataset_Dura_1 = xr.where(((1 - dataset_Bool_1) | (time_AT_1 == time_AT_Index[-1])) & (dataset_Tag_1 >= dura_Thre_1), dataset_Dura_1 + dataset_Tag_1, dataset_Dura_1)
					dataset_Dmax_1 = xr.where(((1 - dataset_Bool_1) | (time_AT_1 == time_AT_Index[-1])) & (dataset_Tag_1 >= dura_Thre_1) & (dataset_Tag_1 > dataset_Dmax_1), dataset_Tag_1, dataset_Dmax_1)
					dataset_Tavg_1 = xr.where(((1 - dataset_Bool_1) | (time_AT_1 == time_AT_Index[-1])) & (dataset_Tag_1 >= dura_Thre_1), dataset_Tavg_1 + dataset_AT_Sum_1, dataset_Tavg_1)
					dataset_Tmax_1 = xr.where(((1 - dataset_Bool_1) | (time_AT_1 == time_AT_Index[-1])) & (dataset_Tag_1 >= dura_Thre_1) & (dataset_AT_Max_1 > dataset_Tmax_1), dataset_AT_Max_1, dataset_Tmax_1)
					dataset_Start_1 = xr.where(((1 - dataset_Bool_1) | (time_AT_1 == time_AT_Index[-1])) & (dataset_Tag_1 >= dura_Thre_1) & (dataset_Start_1 == 0), int(doy_1) - dataset_Tag_1, dataset_Start_1)
					dataset_End_1 = xr.where(((1 - dataset_Bool_1) | (time_AT_1 == time_AT_Index[-1])) & (dataset_Tag_1 >= dura_Thre_1), int(doy_1), dataset_End_1)
					dataset_AT_Sum_1 = xr.where((1 - dataset_Bool_1) & (dataset_Tag_1 >= dura_Thre_1), 0, dataset_AT_Sum_1)
					dataset_Tag_1 = xr.where((1 - dataset_Bool_1) & (dataset_Tag_1 >= dura_Thre_1), 0, dataset_Tag_1)
					print(str(doy_1) + ' of ' +  year_1 + ' is done!')
					text_1.insert(INSERT, str(doy_1) + ' of ' +  year_1 + ' is done!\n')
					text_1.see(END)
					text_1.update_idletasks()
				dataset_HWDR_1 = xr.concat(HWDR_List_1, dim = 'time')
				dataset_Freq_1 = dataset_Freq_1.where(dataset_Freq_1 > 0, drop = False)
				dataset_Dura_1 = dataset_Dura_1.where(dataset_Freq_1 > 0, drop = False)
				dataset_Davg_1 = dataset_Dura_1 / dataset_Freq_1
				dataset_Dmax_1 = dataset_Dmax_1.where(dataset_Freq_1 > 0, drop = False)
				dataset_Tavg_1 = dataset_Tavg_1 / dataset_Dura_1
				dataset_Tmax_1 = dataset_Tmax_1.where(dataset_Freq_1 > 0, drop = False)
				dataset_Start_1 = dataset_Start_1.where(dataset_Freq_1 > 0, drop = False)
				dataset_End_1 = dataset_End_1.where(dataset_Freq_1 > 0, drop = False)
				# 输出结果
				dataset_HWDR_1 = dataset_HWDR_1.rename({'CAT': 'HWDR'})
				dataset_Freq_1 = dataset_Freq_1.rename({'CAT': 'HWF'})
				dataset_Dura_1 = dataset_Dura_1.rename({'CAT': 'HWD'})
				dataset_Davg_1 = dataset_Davg_1.rename({'CAT': 'HWAD'})
				dataset_Dmax_1 = dataset_Dmax_1.rename({'CAT': 'HWMD'})
				dataset_Tavg_1 = dataset_Tavg_1.rename({'CAT': 'HWAT'})
				dataset_Tmax_1 = dataset_Tmax_1.rename({'CAT': 'HWMT'})
				dataset_Start_1 = dataset_Start_1.rename({'CAT': 'HWSD'})
				dataset_End_1 = dataset_End_1.rename({'CAT': 'HWED'})
				dataset_HWR_1 = xr.merge([dataset_Freq_1, dataset_Dura_1, dataset_Davg_1, dataset_Dmax_1, dataset_Tavg_1, dataset_Tmax_1, dataset_Start_1, dataset_End_1])
				time_1 = pd.to_datetime([year_1 + '0101'])
				time_2 = xr.DataArray(time_1, [('time', time_1)])
				dataset_HWR_1['time'] = time_2
				dataset_HWR_1.to_netcdf(outPath_HW_3)
				dataset_HWDR_1.to_netcdf(outPath_HW_4)
				print('Finished!')

# 计算逐日温度或体感温度的界面
def AT_GUI(window_1, text_1):
	tkinter.Label(window_1, text = 'Data source').grid(row = 0, column = 0, padx = 10, pady = 10, sticky = tkinter.W)
	combobox_1 = ttk.Combobox(window_1, values = ['CRU-JRA', 'ERA5', 'GLDAS'], width = 25, state = 'readonly') # 收到消息执行go函数
	combobox_1.current(0)
	combobox_1.grid(row = 0, column = 1, padx = 10, pady = 10, sticky = tkinter.W)

	# ------------------------------------------------------------------
	def SelectPath_1():
		path_ = askdirectory() #使用askdirectory()方法返回文件夹的路径
		if path_ == '':
			inPath_1.get() #当打开文件路径选择框后点击'取消' 输入框会清空路径，所以使用get()方法再获取一次路径
		else:
			# path_ = path_.replace('/', '\\')  # 实际在代码中执行的路径为“\“ 所以替换一下
			inPath_1.set(path_)
	inPath_1 = StringVar()
	inPath_1.set(os.path.abspath('.'))
	tkinter.Label(window_1, text = 'Input directory').grid(row = 0, column = 2, padx = 10, pady = 10, sticky = tkinter.W)
	entry_1 = tkinter.Entry(window_1, textvariable = inPath_1, width = 20, state = 'readonly')
	entry_1.grid(row = 0, column = 3, padx = 10, pady = 10, sticky = tkinter.W)
	tkinter.Button(window_1, text = 'Select', command = SelectPath_1).grid(row = 0, column = 3, padx = 10, pady = 10, sticky = tkinter.E)

	# ------------------------------------------------------------------
	tkinter.Label(window_1, text = 'Statistic').grid(row = 1, column = 0, padx = 10, pady = 10, sticky = tkinter.W)
	combobox_4 = ttk.Combobox(window_1, values = ['Mean', 'Minimum', 'Maximum'], width = 25, state = 'readonly') # 收到消息执行go函数
	combobox_4.current(2)
	combobox_4.grid(row = 1, column = 1, padx = 10, pady = 10, sticky = tkinter.W)

	# ------------------------------------------------------------------
	tkinter.Label(window_1, text = 'Start year').grid(row = 1, column = 2, padx = 10, pady = 10, sticky = tkinter.W)
	years_1 = []
	for year_1 in range(1971, 2021):
		years_1.append(str(year_1))
	combobox_2 = ttk.Combobox(window_1, values = years_1, width = 25, state = 'readonly') # 收到消息执行go函数
	combobox_2.current(0)
	combobox_2.grid(row = 1, column = 3, padx = 10, pady = 10, sticky = tkinter.W)

	# ------------------------------------------------------------------
	tkinter.Label(window_1, text = 'End year').grid(row = 2, column = 0, padx = 10, pady = 10, sticky = tkinter.W)
	combobox_3 = ttk.Combobox(window_1, values = years_1, width = 25, state = 'readonly') # 收到消息执行go函数
	combobox_3.current(len(years_1) - 1)
	combobox_3.grid(row = 2, column = 1, padx = 10, pady = 10, sticky = tkinter.W)

	# ------------------------------------------------------------------
	def SelectPath_2():
		path_ = askdirectory() #使用askdirectory()方法返回文件夹的路径
		if path_ == '':
			outPath_1.get() #当打开文件路径选择框后点击'取消' 输入框会清空路径，所以使用get()方法再获取一次路径
		else:
			# path_ = path_.replace('/', '\\')  # 实际在代码中执行的路径为“\“ 所以替换一下
			outPath_1.set(path_)
	outPath_1 = StringVar()
	outPath_1.set(os.path.abspath('.'))
	tkinter.Label(window_1, text = 'Output directory').grid(row = 2, column = 2, padx = 10, pady = 10, sticky = tkinter.W)
	entry_2 = tkinter.Entry(window_1, textvariable = outPath_1, width = 20, state = 'readonly')
	entry_2.grid(row = 2, column = 3, padx = 10, pady = 10, sticky = tkinter.W)
	tkinter.Button(window_1, text = 'Select', command = SelectPath_2).grid(row = 2, column = 3, padx = 10, pady = 10, sticky = tkinter.E)

	# ------------------------------------------------------------------
	def Cal_APP_Temp():
		data_Source_1 = combobox_1.get()
		inPath_2 = entry_1.get()
		statistic_1 = combobox_4.get()
		start_Year_1 = combobox_2.get()
		end_Year_1 = combobox_3.get()
		outPath_2 = entry_2.get()
		statistic_2 = ''
		if len(statistic_1) == 4:
			statistic_2 = 'tmean'
		else:
			statistic_2 = 't' + statistic_1[0: 3].lower()
		if data_Source_1 == 'CRU-JRA':
			Cal_CRUJRA_AT(inPath_2, start_Year_1, end_Year_1, statistic_2, 'app_temp', outPath_2, text_1)
		elif data_Source_1 == 'ERA5':
			print(inPath_2, start_Year_1, end_Year_1, statistic_2, 'app_temp', outPath_2, text_1)
			Cal_ERA5_AT(inPath_2, start_Year_1, end_Year_1, statistic_2, 'app_temp', outPath_2, text_1)
		elif data_Source_1 == 'GLDAS':
			Cal_GLDAS_AT(inPath_2, start_Year_1, end_Year_1, statistic_2, 'app_temp', outPath_2, text_1)

	def Cal_Air_Temp():
		data_Source_1 = combobox_1.get()
		inPath_2 = entry_1.get()
		statistic_1 = combobox_4.get()
		start_Year_1 = combobox_2.get()
		end_Year_1 = combobox_3.get()
		outPath_2 = entry_2.get()
		statistic_2 = ''
		if len(statistic_1) == 4:
			statistic_2 = 'tmean'
		else:
			statistic_2 = 't' + statistic_1[0: 3].lower()
		if data_Source_1 == 'GLDAS':
			Cal_GLDAS_AT(inPath_2, start_Year_1, end_Year_1, statistic_2, 'air_temp', outPath_2, text_1)
		elif data_Source_1 == 'CRU-JRA':
			Cal_CRUJRA_AT(inPath_2, start_Year_1, end_Year_1, statistic_2, 'air_temp', outPath_2, text_1)
		elif data_Source_1 == 'ERA5':
			Cal_ERA5_AT(inPath_2, start_Year_1, end_Year_1, statistic_2, 'air_temp', outPath_2, text_1)
	# tkinter.Button(window_1, text = 'Apparent Temperature', command = lambda: MyThread(Cal_APP_Temp), width = 27, bg = 'green', fg = 'white').grid(row = 3, column = 2, padx = 10, pady = 10, sticky = tkinter.E)
	tkinter.Button(window_1, text = 'Daily Temperature', command = lambda: MyThread(Cal_Air_Temp), width = 27, bg = 'green', fg = 'white').grid(row = 3, column = 3, padx = 10, pady = 10, sticky = tkinter.E)

# 计算逐日温度或体感温度参考的界面
def AT_Ref_GUI(window_1, text_1):
	def SelectPath_1():
		path_ = askdirectory() #使用askdirectory()方法返回文件夹的路径
		if path_ == '':
			inPath_1.get() #当打开文件路径选择框后点击'取消' 输入框会清空路径，所以使用get()方法再获取一次路径
		else:
			path_ = path_.replace('/', '\\')  # 实际在代码中执行的路径为“\“ 所以替换一下
			inPath_1.set(path_)
			if 'CMIP6' in path_:
				combobox_1.current(50)
				combobox_2.current(len(years_1) - 1)
			else:
				combobox_1.current(0)
				combobox_2.current(49)				
			outPath_1 = os.path.join(os.path.dirname(path_), '2_Temp_Ref')
			outPath_2.set(outPath_1)

	inPath_1 = StringVar()
	inPath_1.set(os.path.abspath('.'))
	tkinter.Label(window_1, text = 'Input directory').grid(row = 4, column = 0, padx = 10, pady = 10, sticky = tkinter.W)
	entry_1 = tkinter.Entry(window_1, textvariable = inPath_1, width = 20, state = 'readonly')
	entry_1.grid(row = 4, column = 1, padx = 10, pady = 10, sticky = tkinter.W)
	tkinter.Button(window_1, text = 'Select', command = SelectPath_1).grid(row = 4, column = 1, padx = 10, pady = 10, sticky = tkinter.E)

	# ------------------------------------------------------------------
	outPath_2 = StringVar()
	tkinter.Label(window_1, text = 'Output directory').grid(row = 4, column = 2, padx = 10, pady = 10, sticky = tkinter.W)
	entry_2 = tkinter.Entry(window_1, textvariable = outPath_2, width = 27, state = 'readonly')
	entry_2.grid(row = 4, column = 3, padx = 10, pady = 10, sticky = tkinter.W)

	# ------------------------------------------------------------------
	tkinter.Label(window_1, text = 'Start year').grid(row = 5, column = 0, padx = 10, pady = 10, sticky = tkinter.W)
	years_1 = []
	for year_1 in range(1971, 2101):
		years_1.append(str(year_1))
	combobox_1 = ttk.Combobox(window_1, values = years_1, width = 25, state = 'readonly') # 收到消息执行go函数
	combobox_1.current(0)
	combobox_1.grid(row = 5, column = 1, padx = 10, pady = 10, sticky = tkinter.W)

	# ------------------------------------------------------------------
	tkinter.Label(window_1, text = 'End year').grid(row = 5, column = 2, padx = 10, pady = 10, sticky = tkinter.W)
	combobox_2 = ttk.Combobox(window_1, values = years_1, width = 25, state = 'readonly') # 收到消息执行go函数
	combobox_2.current(len(years_1) - 1)
	combobox_2.grid(row = 5, column = 3, padx = 10, pady = 10, sticky = tkinter.W)

	# ------------------------------------------------------------------
	tkinter.Label(window_1, text = 'Time window').grid(row = 6, column = 0, padx = 10, pady = 10, sticky = tkinter.W)
	windows_1 = []
	for window_2 in range(4, 11):
		windows_1.append(str(window_2))
	combobox_3 = ttk.Combobox(window_1, values = windows_1, width = 25, state = 'readonly') # 收到消息执行go函数
	combobox_3.current(3)
	combobox_3.grid(row = 6, column = 1, padx = 10, pady = 10, sticky = tkinter.W)

	# ------------------------------------------------------------------
	tkinter.Label(window_1, text = 'Percentile threshold').grid(row = 6, column = 2, padx = 10, pady = 10, sticky = tkinter.W)
	percentiles_1 = []
	for percentile_1 in range(85, 96):
		percentiles_1.append(str(percentile_1))
	combobox_4 = ttk.Combobox(window_1, values = percentiles_1, width = 25, state = 'readonly') # 收到消息执行go函数
	combobox_4.current(5)
	combobox_4.grid(row = 6, column = 3, padx = 10, pady = 10, sticky = tkinter.W)

	# ------------------------------------------------------------------
	def Cal_Ref_3():
		inPath_2 = entry_1.get()
		start_Year_1 = combobox_1.get()
		end_Year_1 = combobox_2.get()
		time_window_1 = combobox_3.get()
		percentiles_1 = combobox_4.get()
		print(inPath_2, start_Year_1, end_Year_1, time_window_1, percentiles_1)
		Cal_Ref_1(inPath_2, start_Year_1, end_Year_1, time_window_1, percentiles_1, text_1)
		# Cal_Ref_2(inPath_2, start_Year_1, end_Year_1, 10, time_window_1, percentiles_1, text_1)

	tkinter.Button(window_1, text = 'High Temperature Threshold', command = lambda: MyThread(Cal_Ref_3), width = 27, bg = 'green', fg = 'white').grid(row = 7, column = 3, padx = 10, pady = 10, sticky = tkinter.E)

# 计算逐年高温热浪的界面
def HW_GUI(window_1, text_1):
	tkinter.Label(window_1, text = 'Percentile threshold').grid(row = 9, column = 0, padx = 10, pady = 10, sticky = tkinter.W)
	combobox_1 = ttk.Combobox(window_1, width = 25, state = 'readonly') # 收到消息执行go函数
	combobox_1.grid(row = 9, column = 1, padx = 10, pady = 10, sticky = tkinter.W)

	# ------------------------------------------------------------------
	def SelectPath_1():
		path_ = askdirectory() #使用askdirectory()方法返回文件夹的路径
		if path_ == '':
			inPath_1.get() #当打开文件路径选择框后点击'取消' 输入框会清空路径，所以使用get()方法再获取一次路径
		else:
			path_ = path_.replace('/', '\\')  # 实际在代码中执行的路径为“\“ 所以替换一下
			inPath_1.set(path_)
			outPath_1 = os.path.join(os.path.dirname(path_), '2_Temp_Ref')
			combobox_1['values'] = os.listdir(outPath_1)
			combobox_1.current(0)
			outPath_2 = os.path.join(os.path.dirname(path_), '3_HWE_Annual')
			outPath_3.set(outPath_2)			

	inPath_1 = StringVar()
	inPath_1.set(os.path.abspath('.'))
	tkinter.Label(window_1, text = 'Input directory').grid(row = 8, column = 0, padx = 10, pady = 10, sticky = tkinter.W)
	entry_1 = tkinter.Entry(window_1, textvariable = inPath_1, width = 20, state = 'readonly')
	entry_1.grid(row = 8, column = 1, padx = 10, pady = 10, sticky = tkinter.W)
	tkinter.Button(window_1, text = 'Select', command = SelectPath_1).grid(row = 8, column = 1, padx = 10, pady = 10, sticky = tkinter.E)

	# ------------------------------------------------------------------
	outPath_3 = StringVar()
	tkinter.Label(window_1, text = 'Output directory').grid(row = 8, column = 2, padx = 10, pady = 10, sticky = tkinter.W)
	entry_2 = tkinter.Entry(window_1, textvariable = outPath_3, width = 27, state = 'readonly')
	entry_2.grid(row = 8, column = 3, padx = 10, pady = 10, sticky = tkinter.W)

	# ------------------------------------------------------------------
	tkinter.Label(window_1, text = 'Constant threshold').grid(row = 9, column = 2, padx = 10, pady = 10, sticky = tkinter.W)
	constants_1 = []
	for constant_1 in range(25, 41):
		constants_1.append(str(constant_1))
	combobox_2 = ttk.Combobox(window_1, values = constants_1, width = 25, state = 'readonly') # 收到消息执行go函数
	combobox_2.current(10)
	combobox_2.grid(row = 9, column = 3, padx = 10, pady = 10, sticky = tkinter.W)

	# ------------------------------------------------------------------
	tkinter.Label(window_1, text = 'Duration threshold').grid(row = 10, column = 0, padx = 10, pady = 10, sticky = tkinter.W)
	durations_1 = []
	for duration_1 in range(3, 11):
		durations_1.append(str(duration_1))
	combobox_3 = ttk.Combobox(window_1, values = durations_1, width = 25, state = 'readonly') # 收到消息执行go函数
	combobox_3.current(0)
	combobox_3.grid(row = 10, column = 1, padx = 10, pady = 10, sticky = tkinter.W)

	# ------------------------------------------------------------------
	def Cal_HW_2():
		inPath_2 = entry_1.get()
		outPath_3 = entry_2.get()
		filename_Ref_1 = combobox_1.get()
		constant_2 = combobox_2.get()
		duration_2 = combobox_3.get()
		Cal_HW_1(inPath_2, filename_Ref_1, int(constant_2), int(duration_2), text_1)

	def End_Program(): 
		raise SystemExit 
		sys.exit()

	tkinter.Button(window_1, text = 'Cancel', command = End_Program, width = 27, bg = 'red', fg = 'white').grid(row = 10, column = 2, padx = 10, pady = 10, sticky = tkinter.E)
	tkinter.Button(window_1, text = 'Heat wave', command = lambda: MyThread(Cal_HW_2), width = 27, bg = 'green', fg = 'white').grid(row = 10, column = 3, padx = 10, pady = 10, sticky = tkinter.E)

def main():
	# ***************************************************************************************************************
	# ***************************************************************************************************************
	# 数据预处理

	# ------------------------------------------------------------------
	# # 判断GLDAS数据是否完整下载
	# inPath_NC_1 = r'G:\GLDAS'
	# date_Start_1 = date(2001, 1, 1)   # start date
	# date_End_1 = date(2020, 12, 31)   # end date
	# date_List = pd.date_range(date_Start_1, date_End_1, freq = '3h')
	# for date_1 in date_List:
	# 	filename_Hour_1 = 'GLDAS_NOAH025_3H.A' + date_1.strftime('%Y%m%d.%H') + '00.021.nc4.SUB.nc4'
	# 	inPath_Hour_1 = os.path.join(inPath_NC_1, filename_Hour_1)
	# 	if os.path.exists(inPath_Hour_1):
	# 		pass
	# 	else:
	# 		print(inPath_Hour_1 + ' does not exists.')

	# ------------------------------------------------------------------
	# # 将未来气候数据分割为逐年数据
	# inPath_NC_1 = r'E:\5_meteorology\global_tmax_2021-2100\cnrm_esm2_1_ssp1_2_6'
	# outPath_NC_1 = r'J:\\2_personal\1_papers\6_heat wave variation\2_data\CMIP6'
	# Get_Yearly_Data(inPath_NC_1, outPath_NC_1)

	# ------------------------------------------------------------------
	# # 数据标准化：截取、滚动和重采样
	# inPath_AT_1 = r'J:\\2_personal\1_papers\6_heat wave variation\2_data\ERA5\air_temp\tmax\2_Temp_Ref'
	# outPath_AT_1 = r'J:\\2_personal\1_papers\6_heat wave variation\2_data\CMIP6\BCC-CSM2-MR_ssp126\2_Temp_Ref'
	# Standardize(inPath_AT_1, 0.5, outPath_AT_1)

	# ------------------------------------------------------------------
	# # 将ERA5阈值转换为未来数据的格式
	# inPath_ref_1 = r'J:\2_personal\1_papers\6_heat wave variation\2_data\CMIP6\GFDL-ESM4_ssp126\1_Temp_Daily\gfdl-esm4_ssp126_air_temp_daily_tmax_2021.nc'
	# inPath_old_1 = r'J:\2_personal\1_papers\6_heat wave variation\2_data\GLDAS\air_temp\tmax\2_Temp_Ref\gldas_air_temp_daily_tmax_90_7_1971-2020.nc'
	# outPath_new_1 = r'J:\2_personal\1_papers\6_heat wave variation\2_data\CMIP6\GFDL-ESM4_ssp126\2_Temp_Ref\gldas_air_temp_daily_tmax_90_7_1971-2020.nc'
	# Interpolate(inPath_ref_1, inPath_old_1, outPath_new_1)

	# ***************************************************************************************************************
	# ***************************************************************************************************************
	# 计算逐日气温和逐年高温热浪
	
	window_1 = tkinter.Tk() #构造窗体
	window_1.title('Global Heat Wave Toolbox')#标题
	window_1.geometry('1050x530')#800宽度，800高度，x,y坐标，左上角

	text_1 = tkinter.Text(window_1, width = 30)
	text_1.place(x = 810, y = 70, anchor = 'nw')
	scrollbar_1 = tkinter.Scrollbar()
	scrollbar_1.place(x = 1006, y = 72, height = 313, anchor = 'nw')
	scrollbar_1.config(command = text_1.yview)
	text_1.config(yscrollcommand = scrollbar_1.set)

	AT_GUI(window_1, text_1)
	AT_Ref_GUI(window_1, text_1)
	HW_GUI(window_1, text_1)

	window_1.mainloop() #进入消息循环机制

main()
