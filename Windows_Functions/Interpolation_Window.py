from PyQt5.QtWidgets import QWidget
from Windows_Forms.Interpolation import Ui_Form
from PyQt5.QtWidgets import QMessageBox,QTableWidgetItem,QHeaderView
from PyQt5.QtWidgets import QFileDialog
from Methods.Interpolation import Interpolate_Way
from Common_Thread import Work_Intepolation_Thread
from Common import CommonVar

# todo Interpolation_Form thread
class Interpolation_Form(QWidget, Ui_Form):
    def __init__(self):
        super(QWidget, self).__init__()
        self.setupUi(self)
        self.file_Array=[]# 当前展示的数据数组
        self.Save_path=[]#当前展示的保存数据路径
        self.FileList=[]#当前的所有文件
        self.ShapeFileList = []  # 当前的掩膜所有文件
        self.lat_min=''
        self.lat_max=''
        self.lon_min=''
        self.lon_max=''
        self.groupBox_3.setVisible(False)
        Common_Var=CommonVar()
        self.comboBox.addItems(Common_Var.Interpolation_Method)
        self.comboBox_2.addItems(Common_Var.Downsacaling_Method)
        self.pushButton.clicked.connect(self.OpenFile)  # TODO 点击 Open File
        self.pushButton_2.clicked.connect(self.SaveFile)  # TODO 点击 Save File
        self.pushButton_3.clicked.connect(self.Run_Interpolation) # TODO 点击 RUN Interpolation Procedure
        self.pushButton_8.clicked.connect(self.Switch_Help)
        self.pushButton_9.clicked.connect(self.OpenMaskFile)
        self.lineEdit_9.editingFinished.connect(self.text_changed)
        self.lineEdit_10.editingFinished.connect( self.text_changed)
        self.lineEdit_11.editingFinished.connect(self.text_changed)
        self.lineEdit_12.editingFinished.connect(self.text_changed)

    # TODO Open files
    def OpenFile(self):
        try:
            Open_Files, _ = QFileDialog.getOpenFileNames(self, 'Select Files', "././Data", "All Files(*)")
            if len(self.FileList)==0:
                 self.FileList=Open_Files
            else:
                self.FileList.extend(Open_Files)
            self.tableWidget.clear()
            self.tableWidget.setRowCount(len(self.FileList))
            self.tableWidget.setColumnCount(1)
            self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.tableWidget.setHorizontalHeaderLabels(["File name"])
            for i in range(len(self.FileList)):
                self.tableWidget.setItem(i, 0, QTableWidgetItem(self.FileList[i]))
            if len(Open_Files)!=0:
                # todo 如果还没有展示数据，则默认展示第一个
                self.lineEdit.setText(self.FileList[0].rpartition('/')[0])
                self.IW = Interpolate_Way(self.FileList[0])
                self.lat_min,self.lat_max,self.lon_min,self.lon_max,lat_len,lon_len=self.IW.GetResolution()
                self.lineEdit_4.setText(str(round((self.lon_max-self.lon_min)/lon_len,2)))
                self.lineEdit_3.setText(str(round((self.lat_max-self.lat_min)/lat_len,2)))
                self.lineEdit_9.setText(str(self.lat_min))
                self.lineEdit_10.setText(str(self.lat_max))
                self.lineEdit_11.setText(str(self.lon_min))
                self.lineEdit_12.setText(str(self.lon_max))
            else:
                QMessageBox.warning(self,"Warning","Files are not selected!")
        except FileNotFoundError:
            QMessageBox.critical(self,"Error","FileNotFoundError successfully handled")
        return "success"

    def OpenMaskFile(self):
        # todo 添加掩膜文件，若不添加，则默认全局插值
        try:
            Open_Files, _ = QFileDialog.getOpenFileNames(self, 'Select Files', "././Data", "All Files(*)")
            if len(self.ShapeFileList)==0:
                 self.ShapeFileList=Open_Files
                 self.lineEdit_8.setText(self.ShapeFileList[0])
            else:
                QMessageBox.warning(self,"Warning","Files are not selected!")
        except FileNotFoundError:
            QMessageBox.critical(self,"Error","FileNotFoundError successfully handled")
        return "success"
    def text_changed(self):
         # 判断是哪个控件触发了事件
        sender=self.sender()
        if sender==self.lineEdit_9:
            if int(self.lineEdit_9.text())<int(self.lat_min):
                QMessageBox.warning(self,"Error","The values defined should not be less than Minimum latitude({})".format(str(self.lat_min)))
                self.lineEdit_9.setText(str(self.lat_min))
        elif sender==self.lineEdit_10:
             if int(self.lineEdit_10.text())>int(self.lat_max):
                QMessageBox.warning(self,"Error","The values defined should not be large than Maximum latitude({})".format(str(self.lat_max)))
                self.lineEdit_10.setText(str(self.lat_max))
        elif sender==self.lineEdit_11:
             if int(self.lineEdit_11.text())<int(self.lon_min):
                QMessageBox.warning(self,"Error","The values defined should not be less than Minimum longitude({})".format(str(self.lon_min)))
                self.lineEdit_11.setText(str(self.lon_min))
        elif sender==self.lineEdit_12:
             if int(self.lineEdit_12.text())>int(self.lon_max):
                QMessageBox.warning(self,"Error","The values defined should not be large than Maximum longitude({})".format(str(self.lon_max)))
                self.lineEdit_12.setText(str(self.lon_max))
    # TODO Save files
    def SaveFile(self):
        self.Save_path = QFileDialog.getExistingDirectory(self, "choose directory", "././Data")
        self.lineEdit_2.setText(self.Save_path)
    def Run_Interpolation(self):
        self.progressBar.setValue(int(0))
        self.WorkerThread1 = Work_Intepolation_Thread(self.ShapeFileList,self.FileList,self.Save_path,self.comboBox.currentText(),self.lineEdit_4.text(),self.lineEdit_3.text(),self.lineEdit_9.text(),self.lineEdit_10.text(),self.lineEdit_11.text(),self.lineEdit_12.text())
        self.WorkerThread1.progressUpdated.connect(self.updateProgress)# todo 使用 self.workerThread.start()效果是一样的
        self.WorkerThread1.start()
    def startCalculation(self):
        if not self.workerThread.isRunning():
            self.workerThread.start()
    def updateProgress(self, progress):
        self.progressBar.setValue(progress)
    def Switch_Help(self):
        self.groupBox_3.setVisible( not self.groupBox_3.isVisible())


# if __name__ == '__main__':
#     app = QApplication([])  # 在 QApplication 方法中使用，创建应用程序对象
#     # app.setWindowIcon(QIcon("Images/Earth.ico"))# 为程序设置图标
#     myWin = Interpolation_Form()  # 实例化 MyMainWindow 类，创建主窗口
#     myWin.setWindowIcon(QIcon("Images/Earth.ico"))
#     myWin.show()  # 在桌面显示控件 myWin
#     app.exec_()


