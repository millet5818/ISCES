import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from xclim.indices.generic import threshold_count,compare
from xclim.indices import run_length as rl
import xarray

df=pd.read_csv('D:\zhaozheng\projects\Global Risk\SBAS_INSAR\ZZ\Deformation\Deformation_all_Indicators_Fileter.csv')
df1=df.drop_duplicates(subset=['Duration','Frequency','Amplitude'], keep='first')
df1.to_csv('D:\zhaozheng\projects\Global Risk\SBAS_INSAR\ZZ\Deformation\Deformation_all_Indicators_Fileter1.csv')



# TODO  CE_Duration复合事件开始到结束的持续总天数，（Duration）,称为，复合事件发生的总天数
def CE_Duration(cond):
    max_l = rl.resample_and_rl(
        cond,
        True,
        rl.windowed_run_count,
        window=1,
        freq='YS',
    )
    # todo 这里只是把不想看到的隐藏了而已
    out = max_l.where(max_l >= 1, np.nan)
    out.attrs["units"] = 'd'
    return out
# TODO  CE_Frequency 复合事件开始到结束的持续总发生数量，（Frequency），称为复合事件一年发生的次数
def CE_Frequency(cond):
    max_l = rl.resample_and_rl(
        cond,
        True,
        rl.windowed_run_events,
        window=1,
        freq='YS',
    )
    # todo 这里只是把不想看到的隐藏了而已
    out = max_l.where(max_l >= 1, np.nan)
    out.attrs["units"] = ''
    return out
# TODO CE_Amplitude 复合事件中发生持续天数最长的天数（amplitude）称为波幅
def CE_Amplitude(cond):
    max_l = rl.resample_and_rl(
        cond,
        True,
        rl.rle_statistics,
        reducer='max',
        window=1,
        freq='YS',
    )
    # todo 这里只是把不想看到的隐藏了而已
    out = max_l.where(max_l >= 1, np.nan)
    out.attrs["units"] = 'd'
    return out




# TODO CE_Start 第一起复合事件的开始时间
def CE_Start(cond):
    out = cond.resample(time='YS').map(
        rl.first_run,
        window=1,
        dim="time",
        coord="dayofyear",
    )
    out.attrs["units"] = ""
    return out

# TODO CE_End 最后一起复合事件的结束时间
def CE_End(cond):
    out: xarray.DataArray = cond.resample(time='YS').map(
        rl.last_run,
        window=1,
        dim="time",
        coord="dayofyear",
    )
    out.attrs["units"] = ""
    return out


shp_file ="D:\zhaozheng\projects\Global Risk\SBAS_INSAR\ZZ\Landslides\Landslides.shp"
Pre_file_threshold="E:\CE_DATA\Data_Processing\Average_Mode\Threshold\CMIP6\Pr_Threshold.nc"
Tem_file_threshold="E:\CE_DATA\Data_Processing\Average_Mode\Threshold\CMIP6\Tem_Threshold.nc"



Times=np.arange(2015,2022,1)
Nums=np.array([60,63,62,21,65,56,54,55,66,67,59,68,58,57,53,52,25,5,6,4,49,51,48,69,3,30,1,2,96,70,73,74,75,72,71,76,
          97,79,77,81,80,41,42,43,44,78,82,38,83,84,37,85,86,95,94,87,90,16,89,91,92,29,93,88])


Colonms_Name=['Points','Year','Deformation_Rate','Date_End','Index','Pre_Threshold','Tem_Threhsold','Duration','Frequency','Amplitude','SD','ED']
 # 提取最大值对应的日期
data_array=np.zeros((7*64,12)).astype(str)
for t in range(len(Times)):
    filename=f"D:\zhaozheng\projects\Global Risk\SBAS_INSAR\ZZ\Deformation\Deformation_all_test_{Times[t]}.csv"
    defor_data=pd.read_csv(filename,parse_dates=['Times'])
    max_values = defor_data.idxmax(axis=0)[1:]  # idxmax返回每列最大值的索引
    max_dates = defor_data.iloc[max_values, :].Times
    max_dates_index=max_dates.index
    max_dates_values=max_dates.values
    for n in range(len(Nums)):
        datasss=defor_data[str(Nums[n])]
        data_array[t+n*7,0]=Nums[n]
        data_array[t+n*7,1]=Times[t]
        data_array[t+n*7,2]=datasss[max_dates_index[n]]
        data_array[t+n*7,3]=pd.to_datetime(max_dates_values[max_dates_index[n]]).strftime('%Y-%m-%d')
        data_array[t+n*7,4]=max_dates_index[n]

shapefile = gpd.read_file(shp_file)
Pre_Thre=xr.open_dataset(Pre_file_threshold).Threshold
Tem_Thre=xr.open_dataset(Tem_file_threshold).Threshold
for t in range(len(Times)):
    print(Times[t])
    Pre_file=f"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Pre\CMIP6\Pr_{Times[t]}.nc"
    Tem_file=f"E:\CE_DATA\Data_Processing\Average_Mode\Daily_Tem\CMIP6\Tem_{Times[t]}.nc"
    Pre_array=xr.open_dataset(Pre_file).pr
    Tem_array=xr.open_dataset(Tem_file).tasmax
    for i in range(len(Nums)):
        Part_Shape=shapefile[shapefile.OBJECTID==Nums[i]]# 获取部分要素
        Boundary=Part_Shape.total_bounds
        aoi_lat = [float(Boundary[1]), float(Boundary[3])]
        aoi_lon = [float(Boundary[0]), float(Boundary[2])]
        Pre_Threshold_value = Pre_Thre.sel(lon=[aoi_lon[0], aoi_lon[1]],lat=[aoi_lat[0], aoi_lat[1]],quantile=0.7,method="nearest").mean(dim=['lat','lon'])
        Tem_Threshold_value = Tem_Thre.sel(lon=[aoi_lon[0], aoi_lon[1]],lat=[aoi_lat[0], aoi_lat[1]],quantile=0.7,method="nearest").mean(dim=['lat','lon'])
        data_array[t+i*7,5]=Pre_Threshold_value.values
        data_array[t+i*7,6]=Tem_Threshold_value.values
        # TODO  开始计算复合事件指标
        Pre_value = Pre_array.sel(lon=[aoi_lon[0], aoi_lon[1]],lat=[aoi_lat[0], aoi_lat[1]],method="nearest").mean(dim=['lat','lon'])
        Tem_value = Tem_array.sel(lon=[aoi_lon[0], aoi_lon[1]],lat=[aoi_lat[0], aoi_lat[1]],method="nearest").mean(dim=['lat','lon'])
        Time_End=data_array[t+i*7,3]

        Pre_value_Clip=Pre_value.sel(time=slice(Time_End))
        Tem_value_Clip=Tem_value.sel(time=slice(Time_End))
        constrain = (">", ">=")
        cond_Pr = compare(Pre_value_Clip, ">", Pre_Threshold_value.values, constrain).astype(int)
        cond_Tem = compare(Tem_value_Clip, ">", Tem_Threshold_value.values, constrain).astype(int)
        mask=np.logical_and(cond_Pr,cond_Tem)
        bbb=CE_Duration(mask)
        ccc=CE_Frequency(mask)
        ddd=CE_Amplitude(mask)
        hhh=CE_Start(mask)
        iii=CE_End(mask)
        data_array[t+i*7,7]=bbb.values[0]
        data_array[t+i*7,8]=ccc.values[0]
        data_array[t+i*7,9]=ddd.values[0]
        data_array[t+i*7,10]=hhh.values[0]
        data_array[t+i*7,11]=iii.values[0]
df=pd.DataFrame(data_array,columns=Colonms_Name)
df.to_csv('D:\zhaozheng\projects\Global Risk\SBAS_INSAR\ZZ\Deformation\Deformation_all_Indicators.csv')

df_fileter=df.dropna()
df.to_csv('D:\zhaozheng\projects\Global Risk\SBAS_INSAR\ZZ\Deformation\Deformation_all_Indicators_Fileter.csv')





