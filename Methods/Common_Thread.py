"""
用于保存进程类
"""
from Files_description import Files_Read
from PyQt5.QtCore import QThread,pyqtSignal
from Extreme_Index_Calucation import Extre_Index
from Compound_Events_Calculation import CE_Index
from Methods.Interpolation import Interpolate_Way


# todo read file thread
class Read_File_Thread(QThread):
    ReadFileFinishSignal=pyqtSignal(list)
    def __init__(self,Selected_FileList):
        super(Read_File_Thread, self).__init__()
        self.Selected_FileList=Selected_FileList
    def run(self):
        self.filesread=Files_Read(self.Selected_FileList)
        Variable,Time_Array=self.filesread.Read_File_Array()
        self.ReadFileFinishSignal.emit([Variable, min(Time_Array).values,max(Time_Array).values])

# todo work thread
class Work_ET_Thread(QThread):
    FinishSignal=pyqtSignal(str)
    # TODO 带参数示例 极端事件计算方法;极端事件指标；需要计算的文件数组;计算的文件变量;保存的文件目录;阈值;时间范围;需要计算的文件;
    def __init__(self, ET_Method,ET_Indicator,variable,save_Directory,Threshod,Time_array,Selected_FileList, ThsholdFile, parent=None):
        super(Work_ET_Thread, self).__init__(parent)
        self.ET_Method = ET_Method
        self.ET_Indicator = ET_Indicator
        self.variable=variable
        self.save_Directory=save_Directory
        self.Threshod=Threshod
        self.Time_array=Time_array
        self.Selected_FileList=Selected_FileList
        self.ThsholdFile= ThsholdFile
    def run(self):
        EI = Extre_Index(self.ET_Method,self.ET_Indicator,self.variable,self.save_Directory,self.Threshod,self.Time_array,self.Selected_FileList,self.ThsholdFile)
        #0: 相对阈值(百分位阈值)---传递的参数不同而已; 1:绝对阈值
        self.process_option(self.ET_Indicator,EI)  # 输出：执行操作1
        # self.FinishSignal.emit("success")

    # TODO switch-case 选择函数
    def process_option(self,option,EI):
        actions = {
            0: lambda: EI.Pre_Related(),
            1: lambda: EI.Tem_Related(),
            2: lambda: print("执行操作3"),
        }
        action = actions.get(option, lambda: print("误操作"))
        action()




# todo CE work thread
class Work_CE_Thread(QThread):
    FinishSignal=pyqtSignal(str)
    # 带参数示例
    def __init__(self,Files1,Files2,Var1,Var2,Method,T_Duration,S_Duration,Var1_Threshod,Var2_Threshold,value,Directory_CE , parent=None):
        super(Work_CE_Thread, self).__init__(parent)
        self.Files1=Files1
        self.Files2=Files2
        self.Var1=Var1
        self.Var2=Var2
        self.Method=Method
        self.T_Duration=T_Duration
        self.S_Duration=S_Duration
        self.Var1_Threshod=Var1_Threshod
        self.Var2_Threshod=Var2_Threshold
        self.Directory_CE=Directory_CE
        self.value = value
    def run(self):
        CE = CE_Index(self.Files1,self.Files2,self.Var1,self.Var2,self.Method,self.T_Duration,self.S_Duration,self.Var1_Threshod,self.Var2_Threshod,self.Directory_CE)
        if(self.value==0):
            CE.HighPrecipication_HighTemperatute_D1()
        elif (self.value == 1):
            CE.HighPrecipication_HighTemperatute_D2()
        elif (self.value == 2):
            CE.HighPrecipication_HighTemperatute_D3()
        elif (self.value == 3):
            CE.HighPrecipication_HighTemperatute_D4()
        elif (self.value == 4):
            CE.HighPrecipication_HighTemperatute_F1()
        elif (self.value == 5):
            CE.HighPrecipication_HighTemperatute_F2()
        elif (self.value == 6):
            CE.HighPrecipication_HighTemperatute_F3()
        elif (self.value == 7):
            CE.HighPrecipication_HighTemperatute_F4()
        elif (self.value == 8):
            CE.HighPrecipication_HighTemperatute_F5()
        elif (self.value == 9):
            CE.HighPrecipication_HighTemperatute_F6()
        elif (self.value == 10):
            CE.HighPrecipication_HighTemperatute_MaxLength1()
        elif (self.value == 11):
            CE.HighPrecipication_HighTemperatute_MaxLength2()
        elif (self.value == 12):
            CE.HighPrecipication_HighTemperatute_MaxLength3()
        elif(self.value==13):
            CE.HighPrecipication_HighTemperatute_EN1()
        elif(self.value==14):
            CE.HighPrecipication_HighTemperatute_EN2()
        elif(self.value == 15):
            CE.HighPrecipication_HighTemperatute_EN3()
        elif(self.value==16):
            CE.HighPrecipication_HighTemperatute_RA1()
        elif(self.value==17):
            CE.HighPrecipication_HighTemperatute_RA2()
        elif(self.value==18):
            CE.HighPrecipication_HighTemperatute_RA3()
        self.FinishSignal.emit("success")
# TODO Interpolation work Thread
class Work_Intepolation_Thread(QThread):
     progressUpdated  = pyqtSignal(int)
     def __init__(self,ShapeFileList,FileList,SavePath,Method,x_lon,y_lat,lat_min,lat_max,lon_min,lon_max):
        super(Work_Intepolation_Thread, self).__init__()
        self.ShapeFileList=ShapeFileList
        self.FileList = FileList
        self.SavePath = SavePath
        self.Method=Method
        self.x_lon=x_lon
        self.y_lat=y_lat
        self.lat_min=lat_min
        self.lat_max=lat_max
        self.lon_min=lon_min
        self.lon_max=lon_max
     def run(self):
        for i in range(len(self.FileList)):
            self.IW=Interpolate_Way(self.FileList[i])
            self.IW.Regular_Interpolator(self.ShapeFileList,self.Method,self.SavePath,self.x_lon,self.y_lat,self.lat_min,self.lat_max,self.lon_min,self.lon_max)
            self.progressUpdated.emit(int((i+1)/len(self.FileList)*100))
