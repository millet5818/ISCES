import numpy as np
from PyQt5.QtWidgets import QMainWindow, QMessageBox,QGraphicsScene, QMenu, QAction, QTableWidgetItem,QHeaderView
from Windows_Forms.STICES import Ui_MainWindow
from  Windows_Functions.Interpolation_Window import Interpolation_Form
from PyQt5.QtGui import QStandardItemModel,QStandardItem,QCursor
from PyQt5.QtCore import QPoint,Qt
from Methods.Files import ReadFile
from PyQt5.QtWidgets import QFileDialog
from MapShow import MyFigure
from Extreme_Index_Calucation import Extre_Index
import matplotlib as mpl
import xarray as xr
from Common import CommonVar
import os
from PyQt5.QtGui import QIcon
from Common_Thread import Read_File_Thread,Work_CE_Thread,Work_ET_Thread

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# todo 先判断数据维度，有没有时间
# todo UI thread
class MyMainWindow(QMainWindow, Ui_MainWindow):  # 继承 QMainWindow类和 Ui_MainWindow界面类
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.OpenFile)
        self.tabWidget.setCurrentIndex(0)
        self.tabWidget.currentChanged['int'].connect(self.CurrentTable_Event)
        self.spinBox_4.valueChanged.connect(self.change_combix6)
        self.comboBox_6.currentIndexChanged.connect(self.ChangeIndex)
        self.spinBox_5.valueChanged.connect(self.change_combix6)
        self.comboBox_8.currentIndexChanged.connect(self.ChangeIndex)
        self.progressBar.setVisible(False)
        self.variable=[]# TODO 当前数据的字典
        self.showVariable=[]# TODO 数据的字典
        self.time=[]# TODO 当前展示的数据的时间
        self.file_Array=[]# TODO 当前展示的数据数组
        self.file_path=[]# TODO 当前展示的数据路径
        self.Selected_FileList=[]# TODO 当前选中的所有文件
        self.Time_Array=[]# TODO 当前选中的时间数组
        self.FileList=[]#TODO 当前的所有文件
        self.Funtion_Type=0 # TODO 标记一下函数
        self.Selected_FileList_Array=[]# TODO 记录用于指数计算的数组
        self.Directory_Extreme_Index=[]# TODO 当前记录的极端事件保存路径
        self.Threshod=[]# TODO 百分数阈值
        self.Directory_CE_Index=[]# TODO 选择复合事件保存位置
        self.File1=[]
        self.File2=[]
        self.ThsholdFile=[]
        self.actionInterpolation_2.triggered.connect(self.OpenInterpolation)
        self.tableWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tableWidget.customContextMenuRequested[QPoint].connect(self.listWidgetContext)
        self.tableWidget.cellDoubleClicked.connect(self.File_clicked) # todo 添加双击事件
        self.pushButton_9.clicked.connect(self.AddThsholdFile)
        Common_Var=CommonVar()
        self.comboBox.addItems(Common_Var.EC_Methods_List)
        self.comboBox_2.addItems(CommonVar.EC_Index_List)
        self.comboBox_4.addItems(Common_Var.CE_Index_List)
        #self.comboBox_3.addItems(Common_Var.CE_Methods_List)
        # run extreme_index
        self.pushButton_3.clicked.connect(self.RUN_ExtremeIndex)
        self.pushButton_5.clicked.connect(self.RUN_CES)
        # save Output files location
        self.pushButton_2.clicked.connect(self.Save_Extreme_Location)
        self.pushButton_4.clicked.connect(self.Save_CE_Location)
        self.checkBox.stateChanged.connect(lambda:self.checkboxStateChanged(self.checkBox.isChecked(),0.7))
        self.checkBox_2.stateChanged.connect(lambda:self.checkboxStateChanged(self.checkBox_2.isChecked(),0.8))
        self.checkBox_3.stateChanged.connect(lambda:self.checkboxStateChanged(self.checkBox_3.isChecked(),0.9))
        # self.lineEdit_3.editingFinished.connect(self.Update_UserDefined_Thershold)
        self.lineEdit_8.textChanged.connect(self.Update_Directory_CE_Index)
    def checkboxStateChanged(self, state,param):
        if state:
            print('Checkbox被选中')
            self.Threshod.append(param)
        else:
            print('Checkbox未被选中')
            self.Threshod.remove(param)
    def trigger_actHelp(self):  # 动作 actHelp 触发
        QMessageBox.about(self, "About us",
                          """Spatiotemporal Identification of Compound Events\nZheng Zhao, Hengxing Lan\nLaboratory of Resources and Environmental Information System, LREIS\nInstitute of Geographic Sciences and Natural Resources Research, CAS\nContact: zhaoz@lreis.ac.cn, lanx@lreis.ac.cn""")
        return
    def OpenFile(self):
            # todo 打开选择文件，先默认是逐日数据，一年一个文件
        try:
            Open_Files, _ = QFileDialog.getOpenFileNames(self, 'Select Files', "E:\CE_DATA\Data_Processing\ERA5", "All Files(*)")
            self.FileList=Open_Files if len(self.FileList)==0 else self.FileList.extend(Open_Files)
            self.tableWidget.clear()
            self.tableWidget.setRowCount(len(self.FileList))
            self.tableWidget.setColumnCount(1)
            self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.tableWidget.setHorizontalHeaderLabels(["File name"])
            for i in range(len(self.FileList)):
                self.tableWidget.setItem(i, 0, QTableWidgetItem(self.FileList[i]))
            if len(Open_Files)!=0:
                # todo 如果还没有展示数据，则默认展示第一个
                if len(self.file_path)==0 and (self.FileList[0].split('.')[-1]=="nc"):
                        file_Array,showVariable,time,Variable,Info = ReadFile(self.FileList[0])
                        self.lineEdit.setText(self.FileList[0])
                        self.textBrowser.setText(str(Info))
                        self.variable=Variable
                        self.showVariable=showVariable
                        self.time=time
                        self.file_Array=file_Array
                        self.file_path=self.FileList[0]
            else:
                QMessageBox.warning(self,"Warning","Files are not selected!")
        except FileNotFoundError:
            QMessageBox.critical(self,"Error","FileNotFoundError successfully handled")
        return "success"
    def AddThsholdFile(self):
        print("選擇閾值文件")
        Open_Files, _ = QFileDialog.getOpenFileNames(self, 'Select Files', "E:\CE_DATA\Data_Processing\ERA5", "All Files(*)")
        self.ThsholdFile=Open_Files
        self.lineEdit_10.setText(self.ThsholdFile[0])
    def CurrentTable_Event(self,index):
         #todo 判断当前被点击的标签页事件
        if(index==1): # map
            print(1)
        elif (index==2):# show data table
            if len(self.variable)>0 and self.comboBox_5.currentText()=="":
                self.comboBox_5.addItems(self.variable)
                self.spinBox_4.setMaximum(len(self.time[0]))
                self.spinBox_4.setMinimum(self.time[0][0])
                self.comboBox_6.addItems(self.time[1])
                self.label_15.setText("of/"+str(len(self.time[0])))
                self.comboBox_6.setCurrentText(self.time[1][self.spinBox_4.value()-1])
                #self.ChangeTable()

        elif (index == 3):
            print("draw")
            if len(self.variable) > 0 and self.comboBox_7.currentText() == "":
                self.comboBox_7.addItems(self.variable)
                self.spinBox_5.setMaximum(len(self.time[0]))
                self.spinBox_5.setMinimum(self.time[0][0])
                self.comboBox_8.addItems(self.time[1])
                self.label_19.setText("of/" + str(len(self.time[0])))
                self.comboBox_8.setCurrentText(self.time[1][self.spinBox_5.value() - 1])
                #self.ChangeDraw()
        else:
            print(index)
    def change_combix6(self):
        if (self.tabWidget.currentIndex() == 2):
            self.comboBox_6.setCurrentText(self.time[1][self.spinBox_4.value() - 1])
        elif (self.tabWidget.currentIndex() == 3):
            self.comboBox_8.setCurrentText(self.time[1][self.spinBox_5.value() - 1])
        # self.ChangeTable()
    def ChangeIndex(self,tag):
        if (self.Funtion_Type==0):
            if (self.tabWidget.currentIndex()==2):
                self.spinBox_4.setValue(tag+1)
                self.ChangeTable()
            elif (self.tabWidget.currentIndex() == 3):
                self.spinBox_5.setValue(tag + 1)
                self.ChangeDraw()
    def ChangeDraw(self):
        va = self.DisplayVariables()
        data=self.file_Array[self.spinBox_5.value()-1,:,:]
        cmap = mpl.cm.viridis
        norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
        F1 = MyFigure(width=3, height=3, dpi=300)
        F1.axes1 = F1.fig.add_subplot(111)
        F1.axes1.imshow(data,cmap=cmap)
        F1.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='horizontal')
        self.scene = QGraphicsScene()  # 创建一个场景
        self.scene.addWidget(F1)  # 将图形元素添加到场景中
        self.graphicsView_2.setScene(self.scene)  # 将创建添加到图形视图显示窗口
    def ChangeTable(self):
        va=self.DisplayVariables()

        self.model = QStandardItemModel(self.file_Array.shape[1],
                                        self.file_Array.shape[2])
        self.tableView.setModel(self.model)

        self.progressBar_2.setMaximum(100)
        self.progressBar_2.setMinimum(0)
        #for i in range(self.file.variables[va][1].shape[0]):
        for i in range(101):# 暂时假设只有50行，其他不填充
            p = int(i / 100 * 100)
            self.progressBar_2.setValue(p)
            #for j in range(self.file.variables['data'][1].shape[1]): # 多线程刷新问题，待解决
            for j in range(100):
                self.model.setItem(i,j,QStandardItem(str(self.file_Array[self.spinBox_4.value()-1,i,j])))
    def DisplayVariables(self):
        self.comboBox_5.setCurrentText(self.showVariable)
        self.comboBox_7.setCurrentText(self.showVariable)
        return self.showVariable
    def File_clicked(self, function_Type):
        print('单个项目被双击')
        self.Funtion_Type=1
        index = self.tableWidget.currentIndex().row()
        if(self.FileList[index].split('.')[-1]=="nc"):
            self.file_path=self.FileList[index]
            file_Array,showVariable,time,Variable,Info= ReadFile(self.file_path)
            self.lineEdit.setText(self.file_path)
            self.textBrowser.setText(str(Info))
            self.variable=Variable
            self.time=time
            self.showVariable=showVariable
            self.file_Array=file_Array
            # self.ChangeIndex()
            self.shown_spin_com()
            self.ChangeTable()
            self.ChangeDraw()
        self.Funtion_Type=0
    def shown_spin_com(self):
        self.spinBox_4.setValue(1)
        self.spinBox_5.setValue(1)
        self.comboBox_6.clear()
        self.comboBox_6.addItems(self.time[1])
        self.comboBox_8.clear()
        self.comboBox_8.addItems(self.time[1])
    def listWidgetContext(self, point):
        popMenu = QMenu()
        Delete_action = QAction("Remove Files", popMenu)
        Delete_action.triggered.connect(self.Delete_Files)
        popMenu.addAction(Delete_action)
        Index_Calculation = QAction("Index Calculation", popMenu)
        Index_Calculation.triggered.connect(self.Index_Calculation)
        CE_Event_Calculation=QAction("CE Calculation", popMenu)
        CE_Event_Calculation.triggered.connect(self.CE_Event_Calculation)
        popMenu.addAction(CE_Event_Calculation)
        popMenu.addAction(Index_Calculation)
        popMenu.exec_(QCursor.pos())
    def Update_Directory_CE_Index(self):
        self.Directory_CE_Index=self.lineEdit_8.text()
    def Delete_Files(self):
        #QMessageBox().question(None, "询问", "确认删除？", QMessageBox.Yes|QMessageBox.No, QMessageBox.No)
        File_Array=[]
        for i in self.tableWidget.selectionModel().selection().indexes():
            rowNum = i.row()
            File_Array.append(rowNum)
        File_Array.reverse()
        for i in File_Array:
            self.FileList.pop(i)
            self.tableWidget.removeRow(i)
    def Index_Calculation(self):
        print("用于指数计算")
        self.Selected_FileList=[]
        for i in self.tableWidget.selectedIndexes():
            self.Selected_FileList.append(self.FileList[i.row()])
        self.lineEdit_2.setText(str(len(self.Selected_FileList)))
        self.RFT = Read_File_Thread(self.Selected_FileList)
        self.RFT.ReadFileFinishSignal.connect(self.ReadFile_Process)
        self.RFT.start()
    def CE_Event_Calculation(self):
        print("复合指数计算")
        self.Selected_FileList = []
        Variables=[]
        for i in self.tableWidget.selectedIndexes():
            self.Selected_FileList.append(self.FileList[i.row()])
            file=xr.open_mfdataset(self.FileList[i.row()])
            #if (np.isin(file.data_vars,Variables))==False:
            va=[i for i in file.data_vars]
            Variables.append(va[:])
            file.close()
        Variables_Bool=np.unique(Variables)
        self.Files1, self.Files2=[np.array(self.Selected_FileList)[np.isin(Variables, i).flatten()] for i in Variables_Bool]
        self.lineEdit_5.setText(str(len( self.Files1)))
        self.lineEdit_6.setText(str(len( self.Files2)))
        self.lineEdit_12.setText(Variables_Bool[0])
        self.lineEdit_13.setText(Variables_Bool[1])
    def Save_Extreme_Location(self):
        print("选择保存极端事件的位置")
        self.Directory_Extreme_Index = QFileDialog.getExistingDirectory(self, "choose directory", "E:\CE_DATA\Data_Processing\Process_Results")
        # self.Directory_Extreme_Index=QFileDialog.getSaveFileName(None,'set file name and directory',"","Netcdf(*.nc)")[0]
        self.lineEdit_4.setText(self.Directory_Extreme_Index)
    def Save_CE_Location(self):
        print("选择复合事件的保存位置")
        self.Directory_CE_Index=QFileDialog.getSaveFileName(None,'set file name and directory',"","Netcdf(*.nc)")[0]
        self.lineEdit_8.setText(self.Directory_CE_Index)
    def Update_UserDefined_Thershold(self):
        self.Threshod.append(float(self.lineEdit_3.text())) if self.lineEdit_3.text()!="" else print(1)
    def RUN_ExtremeIndex(self):
        print("运行极端指数")
        ET_Indicator=self.comboBox_2.currentIndex()
        ET_Method=self.comboBox.currentIndex()
        self.Time_Array=np.arange(int(self.spinBox.text()),int(self.spinBox_2.text())+1,1)
        self.Threshod.append(float(self.lineEdit_3.text())) if (self.lineEdit_3.text()!="") and (self.lineEdit_3.text() not in self.Threshod)else print(1)
        # TODO 带参数示例 极端事件计算方法;极端事件指标；计算的文件变量;保存的文件目录;阈值;时间范围;需要计算的文件;
        self.th = Work_ET_Thread(ET_Method, ET_Indicator,self.comboBox_9.currentText(), self.Directory_Extreme_Index, self.Threshod, self.Time_Array,
                         self.Selected_FileList, self.ThsholdFile)
        self.th.FinishSignal.connect(self.Progress_event)
        self.th.start()
        self.progressBar.setVisible(True)

    def RUN_CES(self):
        print("运行复合事件提取")
        value=self.comboBox_4.currentIndex()# TODO 当前复合事件类型
        self.th = Work_CE_Thread(self.Files1, self.Files2, self.lineEdit_12.text(),self.lineEdit_13.text(),self.comboBox_3.currentIndex(),self.lineEdit_7.text(),self.lineEdit_11.text(),self.lineEdit_14.text(),self.lineEdit_15.text(),value,self.Directory_CE_Index)
        self.th.FinishSignal.connect(self.Progress_event)
        self.th.start()
    def Progress_event(self,progress):
        print("更新progress STATUS")
        self.progressBar.setValue(progress)
        QMessageBox.information(self,"Successful","The calculation of EI events have finished !") if progress==100 else print(0)
    def OpenInterpolation(self):
        self.Interpolation_UI = Interpolation_Form()
        self.Interpolation_UI.setWindowIcon(QIcon("Images/Earth.ico"))
        self.Interpolation_UI.show()
    def ReadFile_Process(self,process):
        # self.Selected_FileList_Array= process[0]
        self.lineEdit_16.setText(str(process[1]))
        self.comboBox_9.addItems(process[0])
        self.lineEdit_9.setText(str(process[2]))
        self.spinBox.setMinimum(int(process[1]))
        self.spinBox.setMaximum(int(process[2]))
        self.spinBox_2.setMinimum(int(process[1]))
        self.spinBox_2.setMaximum(int(process[2]))
        self.spinBox.setValue(int(process[1]))
        self.spinBox_2.setValue(int(process[2]))
        QMessageBox.information(self,"Successful","Files have been loaded!")
