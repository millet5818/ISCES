"""
极端指数计算
"""
import warnings
warnings.filterwarnings("ignore")
import xarray as xr
import numpy as np
import pandas as pd
import xclim.indicators
import xclim.indices as xi
import xclim.core.units
from xarray import open_mfdataset
from xclim.core.calendar import percentile_doy,resample_doy
from dask.diagnostics import ProgressBar
# TODO Number of days with temperature below a given percentile and precipitation above a given percentile.
from  xclim.indicators.atmos import cold_and_wet_days
# TODO Number of wet days with daily precipitation over a given percentile
from xclim.indicators.atmos import days_over_precip_doy_thresh,days_over_precip_thresh,maximum_consecutive_wet_days
# TODO Fraction of precipitation due to wet days with daily precipitation over a given percentile:R95C,极端降雨对于降雨的贡献率
from xclim.indicators.atmos import fraction_over_precip_doy_thresh,wet_precip_accumulation,max_n_day_precipitation_amount
# TODO Fraction of precipitation over threshold during wet days
from xclim.indicators.atmos import fraction_over_precip_thresh
from xclim.core.units import convert_units_to,amount2rate
class Extre_Index:
    def __init__(self,Method,EI_Type,Variable,Directory,Threshold,Time_Range,FileNameList,ThsholdFile):
        self.Method=Method
        self.EI_Type=EI_Type
        self.Variable=Variable
        self.Directory=Directory
        self.Threshold=Threshold
        self.Time_Range=Time_Range
        self.FileNameList=FileNameList
        self.Files=open_mfdataset(self.FileNameList,concat_dim="time", chunks={'time': -1,'latitude': 200, 'longitude': 200},combine="nested",data_vars='minimal', coords='minimal', compat='override')
        self.Rows=self.Files[self.Variable].shape[1]
        self.Cols = self.Files[self.Variable].shape[2]
        self.ThsholdFile=ThsholdFile

    def Pre_Related(self):
        "计算与降雨相关的指数,基于相对阈值或绝对阈值"
        # TODO R95BM: 日降雨量高于百分位数阈值的二值图，高于视为极端降雨，置为1，低于置为0的逐日map图
        # TODO R95P: 日降雨量高于百分位数阈值的年累积降雨量，百分位数阈值基于有雨日(>1mm/day)计算
        # TODO R95D: 日降雨量高于百分数的阈值的极端降雨天数
        # TODO R95DM: 日降雨量高于百分数的阈值的连续最长极端降雨天数
        # TODO R95C: 极端降雨贡献率=R95P/年总降雨量
        print("计算与降雨相关的指数,基于相对阈值或绝对阈值")
        self.Files[self.Variable] = convert_units_to(self.Files[self.Variable], "mm/d") if self.Files[self.Variable].attrs["units"]=="kg m-2 s-1" else amount2rate(self.Files[self.Variable], out_units="mm/d")
        # todo 计算历史时期的阈值
        if len(self.ThsholdFile)==0:
            Files_Base=self.Files.where(self.Files.time.dt.year.isin(self.Time_Range),drop=True)
            R_threshold = Files_Base[self.Variable].where(Files_Base[self.Variable] >= 1) # todo 以 wetday 为基准
            # todo 注意cmip是latitude,era5是lat
            Threshold_Base = R_threshold.chunk({"time": len(R_threshold.time), "latitude": 200, "longitude": 200}).quantile(self.Threshold, dim="time", keep_attrs=True)
            aaa=Threshold_Base.rename('Threshold').to_netcdf(self.Directory+"/Historical_Threshold.nc",format='NETCDF4', engine='netcdf4',compute=False)
            with ProgressBar():
                aaa.compute()
            Threshold_Base_File=xr.open_dataset(self.Directory+"/Historical_Threshold.nc")

        else:
            print("加載本地文件")
            Threshold_Base_File=xr.open_dataset(self.ThsholdFile[0])
        # # todo 计算历史时期的R95D
        R95D = days_over_precip_thresh(self.Files[self.Variable], Threshold_Base_File.Threshold)
        bbb=R95D.rename('R95D').to_netcdf(self.Directory+"/R95D.nc", format='NETCDF4', engine='netcdf4',compute=False)
        with ProgressBar():
             bbb.compute()
        # todo 计算历史时期的R95C
        R95C= fraction_over_precip_thresh(self.Files[self.Variable], Threshold_Base_File.Threshold)
        cc=R95C.rename('R95C').to_netcdf(self.Directory+"/R95C.nc", format='NETCDF4', engine='netcdf4',compute=False)
        with ProgressBar():
             cc.compute()
        # todo 计算历史时期的R95P
        R95P=wet_precip_accumulation(pr=self.Files[self.Variable],thresh=Threshold_Base_File.Threshold)
        ddd=R95P.rename('R95P').to_netcdf(self.Directory+"/R95P.nc",format='NETCDF4', engine='netcdf4',compute=False)
        with ProgressBar():
             ddd.compute()
        # todo 计算历史时期的R95DM
        R95DM=maximum_consecutive_wet_days(pr=self.Files[self.Variable],thresh=Threshold_Base_File.Threshold)
        eee=R95DM.rename('R95DM').to_netcdf(self.Directory+"/R95DM.nc",format='NETCDF4', engine='netcdf4',compute=False)
        with ProgressBar():
              eee.compute()



    def Tem_Related(self):
        "计算与温度相关的指数,基于相对阈值或绝对阈值"
        # TODO R95BM: 日降雨量高于百分位数阈值的二值图，高于视为极端降雨，置为1，低于置为0的逐日map图
        # TODO R95P: 日降雨量高于百分位数阈值的年累积降雨量，百分位数阈值基于有雨日(>1mm/day)计算
        # TODO R95D: 日降雨量高于百分数的阈值的极端降雨天数
        # TODO R95DM: 日降雨量高于百分数的阈值的连续最长极端降雨天数
        # TODO R95C: 极端降雨贡献率=R95P/年总降雨量
        print(1)




    def Threshold_Percentile(self):
        print("基于基准期的百分位数的阈值估计")
        self.Files[self.Variable] = xclim.core.units.amount2rate(self.Files[self.Variable], out_units="mm/d")
        wetdays_Array = self.Files[self.Variable].where(self.Files[self.Variable] >= 1)  # TODO 定义大于1mm/day为有雨日
        a=xclim.core.calendar.percentile_doy(wetdays_Array,per=95,window=5)
        #b = wetdays_Array.quantile([0.95], dim='time', keep_attrs=True)
        a_results = xclim.indicators.icclim.R95p(pr='tp',pr_per=a,freq='YS',ds=self.Files)
        # b_results= xclim.indicators.icclim.R95p(self.Files[self.Variable], pr_per=b, freq='YS')
    # TODO 日降雨量高于某一百分数的阈值的极端降雨天数
    def R95D(self):
        RD_Y_data = []
        for i in range(self.Rows):
            print(i)
            RD_N_Y_data = []
            data = self.Files.tp[:, i, :].load()
            # TODO 转换单位
            File_Block = xclim.core.units.amount2rate(data, out_units="mm/d")
            wetdays_Array = File_Block.where(File_Block >= 1)  # TODO 定义大于1mm/day为有雨日
            if np.size(wetdays_Array) >= 1:
                for n in range(len(self.Threshold)):
                    RNT = wetdays_Array.quantile([self.Threshold[n]], dim='time', keep_attrs=True)
                    RD_N_Y = xclim.indicators.icclim.R95p(File_Block, pr_per=RNT, freq='YS')
                    RD_N_Y.coords["time"] = pd.DatetimeIndex(RD_N_Y.coords['time']).year  # R95P 每个分位数的降雨天数
                    RD_N_Y = RD_N_Y.drop('quantile')
                    RD_N_Y_data.append(RD_N_Y)
            else:
                # todo 由于分块之后，可能有些块每年都没雨被检测到，所以我们置为NAN
                PF_75_Y = wetdays_Array.sum(dim='time')
                RD_N_Y_data= [PF_75_Y, PF_75_Y, PF_75_Y]
            D_Series=xr.merge([RD_N_Y_data[i].rename('RD_'+str(self.Threshold[i]*100)+"_Y") for i in range(len(self.Threshold))])
            RD_Y_data.append(D_Series)
            del data
        xr.concat(RD_Y_data, dim='latitude').to_netcdf(self.Directory + "/" +'RD_'+''.join(str(int(i*100)) for i in self.Threshold)+'_Y.nc')
    def R95D2(self):
        RD_Y_data = []
        for i in range( self.Rows):
            print(i)
            RD_N_Y_data = []
            data = self.Files.tp[:, i, :].load()
            # TODO 转换单位
            File_Block = xclim.core.units.amount2rate(data, out_units="mm/d")
            wetdays_Array = File_Block.where(File_Block >= 1)  # TODO 定义大于1mm/day为有雨日
            if np.size(wetdays_Array) >= 1:
                for n in range(len(self.Threshold)):
                    RNT = wetdays_Array.quantile([self.Threshold[n]], dim='time', keep_attrs=True)
                    RD_N_Y = xr.where(File_Block>=RNT,1,0).groupby(File_Block.time.dt.year).sum(dim='time')
                    RD_N_Y.coords["time"] = ("year", RD_N_Y.coords["year"].values)
                    RD_N_Y.swap_dims({"year": "time"})
                    RD_N_Y_data.append(RD_N_Y)
            else:
                # todo 由于分块之后，可能有些块每年都没雨被检测到，所以我们置为NAN
                PF_75_Y = wetdays_Array.sum(dim='time')
                RD_N_Y_data= [PF_75_Y, PF_75_Y, PF_75_Y]
            D_Series=xr.merge([RD_N_Y_data[i].rename('RD2_'+str(self.Threshold[i]*100)+"_Y") for i in range(len(self.Threshold))])
            RD_Y_data.append(D_Series)
            del data
        xr.concat(RD_Y_data, dim='latitude').to_netcdf(self.Directory + "/" +'RD2_'+''.join(str(int(i*100)) for i in self.Threshold)+'_Y.nc')
    # TODO 日降雨量高于某一百分数的阈值的极端降雨百分比
    def R95I(self):
            print("R95I")
            RI_Y_data = []
            for i in range(self.Rows):
                print(i)
                RI_N_Y_data = []
                data = self.Files.tp[:, i, :].load()
                # TODO 转换单位
                File_Block = xclim.core.units.amount2rate(data, out_units="mm/d")
                wetdays_Array = File_Block.where(File_Block >= 1)  # TODO 定义大于1mm/day为有雨日
                if np.size(wetdays_Array) >= 1:
                    for n in range(len(self.Threshold)):
                        RNT = wetdays_Array.quantile([self.Threshold[n]], dim='time', keep_attrs=True)
                        RI_N_Y = xclim.indicators.icclim.R95pTOT(File_Block, pr_per=RNT, freq='YS')  # 降雨百分数---95分位数以上
                        RI_N_Y.coords["time"] = pd.DatetimeIndex(RI_N_Y.coords['time']).year
                        RI_N_Y = RI_N_Y.drop('quantile')
                        RI_N_Y_data.append(RI_N_Y)
                else:
                    # todo 由于分块之后，可能有些块每年都没雨被检测到，所以我们置为NAN
                    PF_75_Y = wetdays_Array.sum(dim='time')
                    RI_N_Y_data = [PF_75_Y, PF_75_Y, PF_75_Y]
                D_Series = xr.merge([RI_N_Y_data[i].rename('RI_' + str(self.Threshold[i] * 100) + "_Y") for i in
                                     range(len(self.Threshold))])
                RI_Y_data.append(D_Series)
                del data
            xr.concat(RI_Y_data, dim='latitude').to_netcdf(self.Directory + "/" + 'RI_'+''.join(str(int(i*100)) for i in self.Threshold)+'_Y.nc')
    # TODO 日降雨量高于某一百分数的阈值的年累计降雨量
    def R95P(self):
        print("R95P")
        RP_Y_data = []
        for i in range(self.Rows):
            print(i)
            RP_N_Y_data = []
            data = self.Files.tp[:, i, :].load()
            # TODO 转换单位
            File_Block = xclim.core.units.amount2rate(data, out_units="mm/d")
            wetdays_Array = File_Block.where(File_Block >= 1)  # TODO 定义大于1mm/day为有雨日
            if np.size(wetdays_Array) >= 1:
                for n in range(len(self.Threshold)):
                    RNT = wetdays_Array.quantile([self.Threshold[n]], dim='time', keep_attrs=True)
                    # R95P
                    RP_N_Y = xr.where(File_Block >= RNT, File_Block, 0).groupby(File_Block.time.dt.year).sum(
                        dim='time')
                    RP_N_Y.coords["time"] = ("year", RP_N_Y.coords["year"].values)
                    RP_N_Y.swap_dims({"year": "time"})
                    RP_N_Y = RP_N_Y.drop('quantile')
                    RP_N_Y_data.append(RP_N_Y)
            else:
                # todo 由于分块之后，可能有些块每年都没雨被检测到，所以我们置为NAN
                PF_75_Y = wetdays_Array.sum(dim='time')
                RP_N_Y_data = [PF_75_Y, PF_75_Y, PF_75_Y]
            D_Series = xr.merge([RP_N_Y_data[i].rename('RP_' + str(self.Threshold[i] * 100) + "_Y") for i in
                                 range(len(self.Threshold))])
            RP_Y_data.append(D_Series)
            del data
        xr.concat(RP_Y_data, dim='latitude').to_netcdf(self.Directory + "/" + 'RP_'+''.join(str(int(i*100)) for i in self.Threshold)+'_Y.nc')
    # TODO 日降雨量高于某一百分数的阈值，置为1，低于置为0的Z逐日Map图
    def R95F(self):
        print("R95F")
        PF_D_D_Distribution = []
        for i in range(self.Rows):
            print(i)
            PF_D_Distribution = []
            data = self.Files.tp[:, i, :].load()
            # TODO 转换单位
            File_Block = xclim.core.units.amount2rate(data, out_units="mm/d")
            wetdays_Array = File_Block.where(File_Block >= 1)  # TODO 定义大于1mm/day为有雨日
            if np.size(wetdays_Array) >= 1:
                for n in range(len(self.Threshold)):
                    RNT = wetdays_Array.quantile([self.Threshold[n]], dim='time', keep_attrs=True)
                    # R95F
                    PF_N_Distribution = xr.where(File_Block >= RNT, 1, 0)
                    PF_N_Distribution = PF_N_Distribution.drop('quantile')
                    PF_D_Distribution.append(PF_N_Distribution)
            else:
                # todo 由于分块之后，可能有些块每年都没雨被检测到，所以我们置为NAN
                PF_75_Distribution = wetdays_Array
                PF_D_Distribution = [PF_75_Distribution, PF_75_Distribution, PF_75_Distribution]

            D_Series = xr.merge([PF_D_Distribution[i].rename('PF_' + str(self.Threshold[i] * 100) + "_Y") for i in
                                 range(len(self.Threshold))])

            PF_D_D_Distribution.append(D_Series)
            del data
            del PF_D_Distribution
        xr.concat(PF_D_D_Distribution, dim='latitude').to_netcdf(self.Directory + "/" + 'PF_'+''.join(str(int(i*100)) for i in self.Threshold)+'_Y.nc')
     # TODO 日降雨量高于某一降雨量，视为极端降雨，置为1，低于置为0的逐日map图
    def R10mmF(self):
        print("R10MMf")
        PF_D_D_Distribution = []
        for i in range(self.Rows):
            print(i)
            PF_D_Distribution = []
            data = self.Files.tp[:, i, :].load()
            # TODO 转换单位
            File_Block = xclim.core.units.amount2rate(data, out_units="mm/d")
            for n in range(len(self.Threshold)):
                # R95F
                PF_5_Distribution = xr.where(File_Block >= self.Threshold[n], 1, 0)
                PF_5_Y = PF_5_Distribution.groupby(File_Block.time.dt.year).sum(dim='time')
                PF_5_Y.coords["time"] = ("year", PF_5_Y.coords["year"].values)
                PF_5_Y.swap_dims({"year": "time"})
                PF_D_Distribution.append(PF_5_Distribution)

            D_Series = xr.merge([PF_D_Distribution[i].rename('R10mmF_' + str(self.Threshold[i] * 100) + "_Y") for i in
                                 range(len(self.Threshold))])

            R_Series = xr.merge([PF_D_Distribution[i].rename('R10mmF_' + str(self.Threshold[i] * 100) + "_Y") for i in
                                 range(len(self.Threshold))])


            PF_D_D_Distribution.append(D_Series)
            del data
        xr.concat(PF_D_D_Distribution, dim='latitude').to_netcdf(self.Directory + "/" + 'RF_'+self.Threshold[0]+'_Y.nc')
    # TODO 日降雨量高于某一降雨量，视为极端降雨的天数
    def R10mmD(self):
        print("R10MMD")
        PF_D_D_Distribution = []
        for i in range(self.Rows):
            print(i)
            PF_D_Distribution = []
            data = self.Files.tp[:, i, :].load()
            # TODO 转换单位
            File_Block = xclim.core.units.amount2rate(data, out_units="mm/d")
            for n in range(len(self.Threshold)):
                PF_5_Distribution = xr.where(File_Block >= self.Threshold[n], 1, 0)
                PF_5_Y = PF_5_Distribution.groupby(File_Block.time.dt.year).sum(dim='time')
                PF_5_Y.coords["time"] = ("year", PF_5_Y.coords["year"].values)
                PF_5_Y.swap_dims({"year": "time"})
                PF_D_Distribution.append(PF_5_Y)
            D_Series = xr.merge([PF_D_Distribution[i].rename('R10MMD_' + str(self.Threshold[i] * 100) + "_Y") for i in
                                 range(len(self.Threshold))])
            PF_D_D_Distribution.append(D_Series)
            del data
        xr.concat(PF_D_D_Distribution, dim='latitude').to_netcdf(self.Directory + "/" + 'RD_'+self.Threshold[0]+'_Y.nc')
    # TODO 日降雨量高于某一降雨量的最长连续天数
    def RDMaxLendth(self):
        print("RDMaxLendth")
    # xclim.indices.maximum_consecutive_wet_days
    # TODO 日降雨量高于某百分位数的最长连续天数

    # TODO 日降雨量多年的某百分位数阈值

    # TODO 极端事件或强降雨事件每次发生的起止时间和事件持续长度duration

    # TODO 日平均温度高于某百分位数的总天数
    # 温度的百分数阈值计算是按照五天的窗口期计算
    def TP90D(self):
        print("DP90D")
        TP_Y_data = []
        for i in range(self.Cols):
            print(i)
            TP_N_Y_data = []
            data = self.Files[self.Variable][:, :, i].load()
            # TODO 转换单位
            File_Block = xclim.core.units.convert_units_to(data, "degC")
            for n in range(len(self.Threshold)):
                #温度的百分数阈值计算是按照五天的窗口期计算
                tas_per = percentile_doy(File_Block, per=int(self.Threshold[n]*100)).sel(percentiles=int(self.Threshold[n]*100))
                # tas_per2
                RD_N_Y = xclim.indicators.atmos.tg90p(File_Block, tas_per=tas_per, freq='YS')
                RD_N_Y.drop('percentiles')
               # TODO 两种方法计算是一样的，多看源码
               # thresh=resample_doy(tas_per,File_Block)
               # RD_N_Y2 = xr.where(File_Block >= thresh, 1, 0).groupby(File_Block.time.dt.year).sum(dim='time')
                RD_N_Y.coords["time"] = pd.DatetimeIndex(RD_N_Y.coords['time']).year
                TP_N_Y_data.append(RD_N_Y)
            D_Series = xr.merge([TP_N_Y_data[i].rename('TPD_' + str(self.Threshold[i] * 100) + "_Y") for i in range(len(self.Threshold))])
            TP_Y_data.append(D_Series)
            del data
        xr.concat(TP_Y_data, dim='longitude').to_netcdf(self.Directory + "/" + 'TPD_' + ''.join(str(int(i * 100)) for i in self.Threshold) + '_Y.nc')
            #Heatdays_Array = File_Block.where(File_Block >= 1)  # TODO 定义大于1mm/day为有雨日

    # TODO 日平均温度高于某百分位数的逐日分布
    def TP90F(self):
        print("TP90F")
        TP_Y_data = []
        for i in range(self.Cols):
            print(i)
            TP_N_Y_data = []
            data = self.Files[self.Variable][:, :, i].load()
            # TODO 转换单位
            File_Block = xclim.core.units.convert_units_to(data, "degC")
            for n in range(len(self.Threshold)):
                # 温度的百分数阈值计算是按照五天的窗口期计算
                tas_per = percentile_doy(File_Block, per=int(self.Threshold[n] * 100)).sel(
                    percentiles=int(self.Threshold[n] * 100))
                # TODO 两种方法计算是一样的，多看源码
                thresh=resample_doy(tas_per,File_Block)
                RD_N_Y2 = xr.where(File_Block >= thresh, 1, 0)
                TP_N_Y_data.append(RD_N_Y2)
            D_Series = xr.merge([TP_N_Y_data[i].rename('TPF_' + str(self.Threshold[i] * 100) + "_Y") for i in
                                 range(len(self.Threshold))])
            TP_Y_data.append(D_Series)
            del data
        xr.concat(TP_Y_data, dim='longitude').to_netcdf(
            self.Directory + "/" + 'TPF_' + ''.join(str(int(i * 100)) for i in self.Threshold) + '_Y.nc')

    # TODO 日平均温度高于某一阈值（默认为25dec）的的连续最长天数
    def TPMaxLength(self):
        print("TPMaxLength")
        print("DP90D")
        TP_Y_data = []
        for i in range(self.Cols):
            print(i)
            TP_N_Y_data = []
            data = self.Files[self.Variable][:, :, i].load()
            # TODO 转换单位
            File_Block = xclim.core.units.convert_units_to(data, "degC")
            for n in range(len(self.Threshold)):
                # 温度的百分数阈值计算是按照五天的窗口期计算
                RD_N_Y = xclim.indices.maximum_consecutive_tx_days(File_Block,thresh=float(self.Threshold[n]),freq='YS')
                # TODO 两种方法计算是一样的，多看源码
                # thresh=resample_doy(tas_per,File_Block)
                # RD_N_Y2 = xr.where(File_Block >= thresh, 1, 0).groupby(File_Block.time.dt.year).sum(dim='time')
                RD_N_Y.coords["time"] = pd.DatetimeIndex(RD_N_Y.coords['time']).year
                TP_N_Y_data.append(RD_N_Y)
            D_Series = xr.merge([TP_N_Y_data[i].rename('TPMaxLength_' + str(self.Threshold[i] * 100) + "_Y") for i in
                                 range(len(self.Threshold))])
            TP_Y_data.append(D_Series)
            del data
        xr.concat(TP_Y_data, dim='longitude').to_netcdf(self.Directory + "/" + 'TPMaxLength_' + self.Threshold[0] + '_Y.nc')
        # Heatdays_Array = File_Block.where(File_Block >= 1)  # TODO 定义大于1mm/day为有雨日

    # TODO 日平均温度高于某百分位数的年平均温度

    # TODO 日平均温度的某百分位数阈值

    # TODO 日平均温度高于某阈值的总天数
    def TP30D(self):
        print("TP30D")
    # xclim.indices.tg_days_above

    # TODO 日平均温度高于某阈值的逐日分布
    # TODO 日平均温度高于某阈值的连续最长天数


    # TODO 日平均温度高于某阈值的年平均温度

    def Pre_Extreme_Index(self):
        print("降雨极端指数计算")
        # TODO Step1:基于每个像素点，每日降水量数据（每日降水量≥ 1mm）进行升序排序,获得n年的日降水事件序列
        # TODO Step2: 取每个像元有雨日的第百分之95分位数作为降雨阈值
        Cols = self.FileList[0][self.Variable].shape[2]
        Rows = self.FileList[0][self.Variable].shape[1]
        PF_Y_data = []
        PF_D_D_Distribution = []
        Files=open_mfdataset(self.FileNameList,concat_dim="time", combine="nested",
                  data_vars='minimal', coords='minimal', compat='override')
        for i in range(Rows):
            PF_D_Distribution = []
            PF_N_Y_data = []
            RP_N_Y_data = []
            RD_N_Y_data = []
            RI_N_Y_data = []
            print(i)
            data=Files[:,i,:].tp
            # TODO 转换单位
            File_Block = xclim.core.units.amount2rate(data, out_units="mm/d")
            wetdays_Array = File_Block.where(File_Block >= 1)  # TODO 定义大于1mm/day为有雨日
            if np.size(wetdays_Array) >= 1:
                RNT = wetdays_Array.quantile([self.Threshold], dim='time', keep_attrs=True)
                for n in range(RNT.shape[0]):
                    PF_N_Distribution = xr.where(File_Block >= RNT[n], 1, 0)
                    PF_N_Y = PF_N_Distribution.groupby(File_Block.time.dt.year).sum(dim='time')
                    PF_N_Y.coords["time"] = ("year", PF_N_Y.coords["year"].values)
                    PF_N_Y.swap_dims({"year": "time"})
                    PF_N_Y = PF_N_Y.drop('quantile')
                    PF_N_Distribution = PF_N_Distribution.drop('quantile')
                    PF_N_Y_data.append(PF_N_Y)
                    PF_D_Distribution.append(PF_N_Distribution)

                    # # R95P
                    # RP_N_Y = xr.where(File_Block >= RNT[n], File_Block, 0).groupby(File_Block.time.dt.year).sum(dim='time')
                    # RP_N_Y.coords["time"]=("year",RP_N_Y.coords["year"].values)
                    # RP_N_Y.swap_dims({"year":"time"})
                    # RP_N_Y=RP_N_Y.drop('quantile')
                    # RP_N_Y_data.append(RP_N_Y)
                    #
                    # # R95D
                    # RD_N_Y = xclim.indicators.icclim.R95p(File_Block, pr_per=RNT[n], freq='YS')
                    # RD_N_Y.coords["time"] = pd.DatetimeIndex(RD_N_Y.coords['time']).year  # R95P 每个分位数的降雨天数
                    # RD_N_Y = RD_N_Y.drop('quantile')
                    # RD_N_Y_data.append(RD_N_Y)
                    #
                    # # R95I
                    # RI_N_Y = xclim.indicators.icclim.R95pTOT(File_Block, pr_per=RNT[n], freq='YS')  # 降雨百分数---95分位数以上
                    # RI_N_Y.coords["time"] = pd.DatetimeIndex(RI_N_Y.coords['time']).year
                    # RI_N_Y = RI_N_Y.drop('quantile')
                    # RI_N_Y_data.append(RI_N_Y)

                    # RMCD Maximum number of consecutive days where the daily precipitation is above a threshold

                # todo mm=5mm
                # # PF_10_Y=xclim.indicators.icclim.R10mm(File_Block,freq='YS')
                # # PF_10_Y.coords["time"] = pd.DatetimeIndex(PF_10_Y.coords['time']).year
                # PF_5_Distribution=xr.where(File_Block >= 5, 1, 0)
                # PF_5_Y = PF_5_Distribution.groupby(File_Block.time.dt.year).sum(dim='time')
                # PF_5_Y.coords["time"] = ("year", PF_5_Y.coords["year"].values)
                # PF_5_Y.swap_dims({"year": "time"})
            else:
                # todo 由于分块之后，可能有些块每年都没雨被检测到，所以我们置为NAN
                PF_75_Distribution = wetdays_Array
                PF_75_Y = wetdays_Array.sum(dim='time')
                PF_N_Y_data = [PF_75_Y, PF_75_Y, PF_75_Y]
                # RP_N_Y_data=PF_N_Y_data
                PF_D_Distribution = [PF_75_Distribution, PF_75_Distribution, PF_75_Distribution]
                # RD_N_Y_data=PF_N_Y_data
                # RI_N_Y_data=PF_N_Y_data
                # PF_5_Distribution=PF_75_Distribution
                # PF_5_Y=PF_75_Y
            F_Series = xr.merge(
                [PF_D_Distribution[0].rename('PF_75_Distribution'), PF_D_Distribution[1].rename('PF_85_Distribution'),
                 PF_D_Distribution[2].rename('PF_95_Distribution')])
            R_Series = xr.merge(
                [PF_N_Y_data[0].rename('PF_75_Y'), PF_N_Y_data[1].rename('PF_85_Y'), PF_N_Y_data[2].rename('PF_95_Y')])
            # F_Series = xr.merge(
            #     [PF_5_Distribution.rename('PF_5_Distribution'), PF_D_Distribution[0].rename('PF_75_Distribution'),
            #      PF_D_Distribution[1].rename('PF_85_Distribution'), PF_D_Distribution[2].rename('PF_95_Distribution')])
            # R_Series = xr.merge(
            #     [PF_5_Y.rename('PF_5_Y'), PF_N_Y_data[0].rename('PF_75_Y'), PF_N_Y_data[1].rename('PF_85_Y'),
            #      PF_N_Y_data[2].rename('PF_95_Y')])
            # P_Series=xr.merge([RP_N_Y_data[0].rename('RP_75_Y'), RP_N_Y_data[1].rename('RP_85_Y'), RP_N_Y_data[2].rename('RP_95_Y')])
            # D_Series=xr.merge([RD_N_Y_data[0].rename('RD_75_Y'), RD_N_Y_data[1].rename('RD_85_Y'), RD_N_Y_data[2].rename('RD_95_Y')])
            # I_Series=xr.merge([RI_N_Y_data[0].rename('RI_75_Y'), RI_N_Y_data[1].rename('RI_85_Y'), RI_N_Y_data[2].rename('RI_95_Y')])
            PF_D_D_Distribution.append(F_Series)
            PF_Y_data.append(R_Series)
            del data
            # RP_Y_data.append(P_Series)
            # RD_Y_data.append(D_Series)
            # RI_Y_data.append(I_Series)
        xr.concat(PF_D_D_Distribution, dim='latitude').to_netcdf(self.Directory + "/" + 'PF_Distribution.nc')
        xr.concat(PF_Y_data, dim='latitude').to_netcdf(self.Directory + "/" + 'PF_Y.nc')
        # xr.concat(RP_Y_data, dim='latitude').to_netcdf(self.Directory + "/" + 'RP_Y.nc')
        # xr.concat(RD_Y_data, dim='latitude').to_netcdf(self.Directory + "/" + 'RD_Y.nc')
        # xr.concat(RI_Y_data, dim='latitude').to_netcdf(self.Directory + "/" + 'RI_Y.nc')

    def Pre_Extreme_Index3(self):
        print("降雨极端指数计算")
        # TODO Step1:基于每个像素点，每日降水量数据（每日降水量≥ 1mm）进行升序排序,获得n年的日降水事件序列
        # TODO Step2: 取每个像元有雨日的第百分之95分位数作为降雨阈值

        Cols = self.FileList[0][self.Variable].shape[2]
        Rows = self.FileList[0][self.Variable].shape[1]

        RP_Y_data = []
        PF_Y_data = []
        RD_Y_data = []
        RI_Y_data = []
        PF_D_D_Distribution = []
        # a = xclim.ensembles.create_ensemble(self.FileNameList)
        # files_Array=[self.FileList[j][self.Variable] for j in range(len(self.FileList))]

        # TODO 分块求解，最后拼接
        for i in range(Rows):
            PF_D_Distribution = []
            PF_N_Y_data = []
            RP_N_Y_data = []
            RD_N_Y_data = []
            RI_N_Y_data = []
            print(i)
            file = [self.FileList[j][self.Variable][:, i:i + 1, :] for j in range(len(self.FileList))]
            # del File_Block["realization"]
            # File_Block = File_Block[["time", "latitude", "longitude"]]
            File_Block = xr.concat(file, dim='time')
            # TODO 转换单位
            File_Block = xclim.core.units.amount2rate(File_Block, out_units="mm/d")
            wetdays_Array = File_Block.where(File_Block >= 1)  # TODO 定义大于1mm/day为有雨日
            if np.size(wetdays_Array) >= 1:
                RNT = wetdays_Array.quantile([0.7, 0.85, 0.95], dim='time', keep_attrs=True)
                for n in range(RNT.shape[0]):
                    PF_N_Distribution = xr.where(File_Block >= RNT[n], 1, 0)
                    PF_N_Y = PF_N_Distribution.groupby(File_Block.time.dt.year).sum(dim='time')
                    PF_N_Y.coords["time"] = ("year", PF_N_Y.coords["year"].values)
                    PF_N_Y.swap_dims({"year": "time"})
                    PF_N_Y = PF_N_Y.drop('quantile')
                    PF_N_Distribution = PF_N_Distribution.drop('quantile')
                    PF_N_Y_data.append(PF_N_Y)
                    PF_D_Distribution.append(PF_N_Distribution)

                    # # R95P
                    # RP_N_Y = xr.where(File_Block >= RNT[n], File_Block, 0).groupby(File_Block.time.dt.year).sum(dim='time')
                    # RP_N_Y.coords["time"]=("year",RP_N_Y.coords["year"].values)
                    # RP_N_Y.swap_dims({"year":"time"})
                    # RP_N_Y=RP_N_Y.drop('quantile')
                    # RP_N_Y_data.append(RP_N_Y)
                    #
                    # # R95D
                    # RD_N_Y = xclim.indicators.icclim.R95p(File_Block, pr_per=RNT[n], freq='YS')
                    # RD_N_Y.coords["time"] = pd.DatetimeIndex(RD_N_Y.coords['time']).year  # R95P 每个分位数的降雨天数
                    # RD_N_Y = RD_N_Y.drop('quantile')
                    # RD_N_Y_data.append(RD_N_Y)
                    #
                    # # R95I
                    # RI_N_Y = xclim.indicators.icclim.R95pTOT(File_Block, pr_per=RNT[n], freq='YS')  # 降雨百分数---95分位数以上
                    # RI_N_Y.coords["time"] = pd.DatetimeIndex(RI_N_Y.coords['time']).year
                    # RI_N_Y = RI_N_Y.drop('quantile')
                    # RI_N_Y_data.append(RI_N_Y)

                    # RMCD Maximum number of consecutive days where the daily precipitation is above a threshold

                # todo mm=5mm
                # # PF_10_Y=xclim.indicators.icclim.R10mm(File_Block,freq='YS')
                # # PF_10_Y.coords["time"] = pd.DatetimeIndex(PF_10_Y.coords['time']).year
                # PF_5_Distribution=xr.where(File_Block >= 5, 1, 0)
                # PF_5_Y = PF_5_Distribution.groupby(File_Block.time.dt.year).sum(dim='time')
                # PF_5_Y.coords["time"] = ("year", PF_5_Y.coords["year"].values)
                # PF_5_Y.swap_dims({"year": "time"})
            else:
                # todo 由于分块之后，可能有些块每年都没雨被检测到，所以我们置为NAN
                PF_75_Distribution = wetdays_Array
                PF_75_Y = wetdays_Array.sum(dim='time')
                PF_N_Y_data = [PF_75_Y, PF_75_Y, PF_75_Y]
                # RP_N_Y_data=PF_N_Y_data
                PF_D_Distribution = [PF_75_Distribution, PF_75_Distribution, PF_75_Distribution]
                # RD_N_Y_data=PF_N_Y_data
                # RI_N_Y_data=PF_N_Y_data
                # PF_5_Distribution=PF_75_Distribution
                # PF_5_Y=PF_75_Y
            F_Series = xr.merge(
                [PF_D_Distribution[0].rename('PF_75_Distribution'), PF_D_Distribution[1].rename('PF_85_Distribution'),
                 PF_D_Distribution[2].rename('PF_95_Distribution')])
            R_Series = xr.merge(
                [PF_N_Y_data[0].rename('PF_7_Y'), PF_N_Y_data[1].rename('PF_85_Y'), PF_N_Y_data[2].rename('PF_95_Y')])
            # F_Series = xr.merge(
            #     [PF_5_Distribution.rename('PF_5_Distribution'), PF_D_Distribution[0].rename('PF_75_Distribution'),
            #      PF_D_Distribution[1].rename('PF_85_Distribution'), PF_D_Distribution[2].rename('PF_95_Distribution')])
            # R_Series = xr.merge(
            #     [PF_5_Y.rename('PF_5_Y'), PF_N_Y_data[0].rename('PF_75_Y'), PF_N_Y_data[1].rename('PF_85_Y'),
            #      PF_N_Y_data[2].rename('PF_95_Y')])
            # P_Series=xr.merge([RP_N_Y_data[0].rename('RP_75_Y'), RP_N_Y_data[1].rename('RP_85_Y'), RP_N_Y_data[2].rename('RP_95_Y')])
            # D_Series=xr.merge([RD_N_Y_data[0].rename('RD_75_Y'), RD_N_Y_data[1].rename('RD_85_Y'), RD_N_Y_data[2].rename('RD_95_Y')])
            # I_Series=xr.merge([RI_N_Y_data[0].rename('RI_75_Y'), RI_N_Y_data[1].rename('RI_85_Y'), RI_N_Y_data[2].rename('RI_95_Y')])
            PF_D_D_Distribution.append(F_Series)
            PF_Y_data.append(R_Series)
            # RP_Y_data.append(P_Series)
            # RD_Y_data.append(D_Series)
            # RI_Y_data.append(I_Series)
        xr.concat(PF_D_D_Distribution, dim='latitude').to_netcdf(self.Directory + "/" + 'PF_Distribution.nc')
        xr.concat(PF_Y_data, dim='latitude').to_netcdf(self.Directory + "/" + 'PF_Y2.nc')
        # xr.concat(RP_Y_data, dim='latitude').to_netcdf(self.Directory + "/" + 'RP_Y.nc')
        # xr.concat(RD_Y_data, dim='latitude').to_netcdf(self.Directory + "/" + 'RD_Y.nc')
        # xr.concat(RI_Y_data, dim='latitude').to_netcdf(self.Directory + "/" + 'RI_Y.nc')

    def Tem_Extrem_Index(self):
        print("极端高温计算")
        Cols = self.FileList[0][self.Variable].shape[2]
        Rows = self.FileList[0][self.Variable].shape[1]
        for i in range(Cols):
            File_Block = xr.concat([self.FileList[j][self.Variable][:, :, i:i + 1] for j in range(len(self.FileList))],
                                   dim='time')
            # TODO 转换单位
            File_Block = xclim.core.units.convert_units_to(File_Block, "degC")

            # TODO 相对阈值方法
            TNT=File_Block.quantile([0.7,0.8,0.9], dim='time', keep_attrs=True)

            Heatdays_Array = File_Block.where(File_Block >= 1)  # TODO 定义大于1mm/day为有雨日

            # todo 日最大温度大于90%的天数
            # xclim.indicators.icclim.TX90p
            # todo 日平均温度大于90%的天数
            # xclim.indicators.atmos.tg90p

        #- 273.15

    def Pre_Extreme_Index2(self):
        print("降雨极端指数计算")
        # TODO Step1:基于每个像素点，每日降水量数据（每日降水量≥ 1mm）进行升序排序,获得n年的日降水事件序列
        # TODO Step2: 取每个像元有雨日的第百分之95分位数作为降雨阈值

        Cols=self.FileList[0][self.Variable].shape[2]
        Rows=self.FileList[0][self.Variable].shape[1]
        R95T_data=[]
        R_Series_data=[]
        #a=xclim.ensembles.create_ensemble(self.FileList)
        # TODO 分块求解，最后拼接
        for i in range(Rows):
            print(i)
            #a=xclim.ensembles.create_ensemble(self.FileList)
            File_Block=xr.concat([self.FileList[j][self.Variable][:,i:i+1,:] for j in range(len(self.FileList))],dim='time')
            #File_Block=xclim.core.units.convert_units_to(File_Block,'mm')# TODO 转换单位
            File_Block=xclim.core.units.amount2rate(File_Block,out_units="mm/d")

            wetdays_Array=File_Block.where(File_Block >= 1) # TODO 定义大于1mm/day为有雨日

            if np.size(wetdays_Array) >= 1:
                R95T = wetdays_Array.quantile(0.95, dim='time', keep_attrs=True)  # 有雨日95阈值

                R95P = File_Block.where(File_Block > R95T).groupby(File_Block.time.dt.year).sum(dim='time')
                R95P.coords["time"]=("year",R95P.coords["year"].values)
                R95P.swap_dims({"year":"time"})
                # R95P.drop_dims("year")
                R95D = (wetdays_Array.groupby(File_Block.time.dt.year).count(dim='time') * 0.95).astype(int)
                R95D.coords["time"]=("year",R95D.coords["year"].values)
                R95D.swap_dims({"year": "time"})
                # 年总降雨量
                R95I = xclim.indices.fraction_over_precip_thresh(File_Block, R95T, thresh='1 mm/day',freq='YS')
                R95I.coords["time"] = pd.DatetimeIndex(R95I.coords['time']).year# Fraction of precipitation due to wet days with daily precipitation over a given percentile.

                #R95T = File_Block.quantile(0.95, dim="time", keep_attrs=True) # 95分位数阈值
                R95D2=xclim.indicators.icclim.R95p(File_Block, pr_per=R95T,freq='YS')
                R95D2.coords["time"] = pd.DatetimeIndex(R95D2.coords['time']).year  # R95P 每个分位数的降雨天数

                R95I2=xclim.indicators.icclim.R95pTOT(File_Block, pr_per=R95T,freq='YS')# 降雨百分数---95分位数以上
                R95I2.coords["time"] = pd.DatetimeIndex(R95I2.coords['time']).year

                #R95T.fillna(0)
            else:
                # todo 由于分块之后，可能有些块每年都没雨被检测到，所以我们置为NAN
                Non_Rain_L = wetdays_Array.sum(dim='time')
                R95T=Non_Rain_L
                # R95T.fillna(0)
                R95P=R95T
                R95D=R95T
                R95I=R95T
                R95D2=R95T
                R95I2=R95T
            R95T.attrs['units'] = 'mm'
            R95P.attrs['units'] = 'mm'
            R95D.attrs['units'] = 'd'
            R95I.attrs['units'] = '%'
            R95D2.attrs['units'] = 'd'
            R95I2.attrs['units'] = '%'

            R_Series = xr.merge([ R95D.rename('R95D'), R95P.rename('R95P'), R95I.rename('R95I'),R95D2.rename('R95D2'),R95I2.rename('R95I2')])
            R95T_data.append(R95T.rename('R95T'))
            R_Series_data.append(R_Series)
        xr.concat(R95T_data,dim='latitude').to_netcdf('2023L.nc')
        xr.concat(R_Series_data,dim='latitude').to_netcdf('2023R.nc')
        print(File_Block)


    def Tem_Extrem_Index(File):
        print("极端高温计算")

        #- 273.15

