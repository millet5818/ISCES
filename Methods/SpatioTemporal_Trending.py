import xarray as xr
from scipy import stats
from pylab import *
mpl.rcParams["font.sans-serif"] = ["Times New Roman"] # TODO  Windows_Functions
mpl.rcParams['font.size'] = 16# 设置全局字体大小


def Cal_Trending(OutputFile_P,OutputFile,data,dims):
    data_var=data[list(data.data_vars.keys())[0]]
    trend = np.zeros((3,data_var.latitude.shape[0], data_var.longitude.shape[0]))
    p_value = np.zeros((3,data_var.latitude.shape[0], data_var.longitude.shape[0]))
    intercept=np.zeros((3))
    r_value=np.zeros((3))
    Threshold=[0.7,0.8,0.9]
    for t in range(len(Threshold)):
        data_var_t=data_var.sel(quantile=Threshold[t])
        for i in range (0,data_var.latitude.shape[0]):
            print(i)
            for j in range (0,data_var.longitude.shape[0]):
                # TODO 什么情况下进行线性回归，同一个像素，百分之六十的年份存在复合事件，则进行趋势检验
                if np.where(data_var_t[:,i,j]>1)[0].shape[0]>int(data_var.time.shape[0]*0.6):
                    trend[t,i,j], intercept[t], r_value[t], p_value[t,i,j], std_err=stats.linregress(np.arange(1941,2024,1),data_var_t[:,i,j])
                else:
                    trend[t, i, j]=np.NAN
                    p_value[t, i, j]=np.NAN
        print(trend)
        print(p_value)

    np.save(OutputFile,trend)
    np.save(OutputFile_P,p_value)

data=xr.open_dataset("E:\CE_DATA\Data_Processing\Process_Results\CE_ERA5\CE_End.nc")
OutputFile="E:\CE_DATA\Data_Processing\Process_Results\CE_ERA5\CE_ED_trend.npy"
OutputFile_P="E:\CE_DATA\Data_Processing\Process_Results\CE_ERA5\CE_ED_p_value.npy"
dims=["longitude","time"]
CE_Frequency_Curves=Cal_Trending(OutputFile_P,OutputFile,data,dims)
