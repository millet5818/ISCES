"""
根据目录读取文件
"""
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import xarray as xr
import netCDF4 as nc
import datetime
import gc


def ReadFile(filePath):
    file=nc.Dataset(filePath)
    Variables=file.variables.keys()# todo 获取变量名
    File_Info="Filename:"+filePath+"\n"
    Info_Array=GetFileInfo(file)
    time_Array=[]
    dtime = nc.num2date(file.variables["time"][:], file.variables["time"].units)
    for m in dtime:
        if (file.variables['time'].units.split(' ')[0] == "days"):
            time_Array.append(m.strftime("%Y-%m-%d"))
        elif (file.variables['time'].units.split(' ')[0] == "hours"):
            time_Array.append(m.strftime('%Y-%m-%d %H'))
    # if (file.variables['time'].units.split(' ')[0]=="days"):
    #     time_Begin = datetime.datetime.strptime(file.variables['time'].units.split(" ")[-1], '%Y-%m-%d')
    #    # time_Array1 = [(time_Begin + datetime.timedelta(days=x)).strftime("%Y-%m-%d") for x in range(0, len(file.variables['time']))]
    #     time = [np.arange(1, len(file.variables['time']) + 1), time_Array]
    # elif (file.variables['time'].units.split(' ')[0]=="hours"):
    #     time_Begin = datetime.datetime.strptime(file.variables['time'].units.split("since ")[1].split(":")[0], '%Y-%m-%d %H')
       # time_Array1 = [(time_Begin + datetime.timedelta(hours=x)).strftime("%Y-%m-%d %H") for x in range(0, len(file.variables['time']))]
        time = [np.arange(1, len(file.variables['time']) + 1), time_Array]
    showVariable,dataset=DisplayVariables(file)
    # file.close()
    del file
    return dataset,showVariable,time,Variables,File_Info+Info_Array
def Read_File_Array(filePath_List):
    # TODO 用于指数计算的数组
    with xr.open_mfdataset(filePath_List,concat_dim="time", combine="nested",data_vars='minimal', coords='minimal', compat='override') as file_list:
        Variable =[j for j in file_list.variables.keys()]
        Time_Array=file_list.time.dt.year
        return file_list, Variable,Time_Array
def Read_File_Array2(filePath_List):
    # TODO 用于指数计算的数组
    Time_Array = []
    Variable = []
    Selected_FileList_Array=[]
    for i in filePath_List:
        file = nc.Dataset(i)
        data=xr.open_dataset(i)
        for j in file.variables.keys():
            if len(file.variables[j].shape) > 1:
                Variable.append(j) # 变量
        time_Array=[]
        dtime = nc.num2date(file.variables["time"][:], file.variables["time"].units)
        for m in dtime:
            if (file.variables['time'].units.split(' ')[0] == "days"):
                 time_Array.append(m.strftime("%Y-%m-%d"))
            elif (file.variables['time'].units.split(' ')[0] == "hours"):
                 time_Array.append(m.strftime('%Y-%m-%d %H'))
        Selected_FileList_Array.append(data)
        Time_Array = np.append(Time_Array, np.asarray(time_Array))
        del data
        del file
        gc.collect()
    return Selected_FileList_Array, Variable, Time_Array
def TimeDifference(target_time,format_pattern,cur_time):
    difference = (datetime.datetime.strptime(target_time, format_pattern) - datetime.datetime.strptime(cur_time, format_pattern))
    if difference.days < 0:
        print(target_time, '在当前时间之前')
        return 1
    else:
        print(target_time, '在当前时间之后')
        return 2
def GetFileInfo(file):
    Variables = file.variables.keys()  # todo 获取变量名
    Info=str(file)
    for i in Variables:
        Info=Info+"\n"+i+":"+str(file.variables[i])+"\n"
    return Info
def DisplayVariables(file):
    for i in file.variables.keys():
        if len(file.variables[i].shape) > 1:
            return i,file.variables[i]
            # return i, file.variables[i][:].data
#ReadFile('../Data/AT_1_mean_TD_C_0.25_2006.nc')
