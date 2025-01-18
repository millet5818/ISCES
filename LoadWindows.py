import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from Windows_Functions.ITICES_Window import MyMainWindow
# todo 先判断数据维度，有没有时间
if __name__ == '__main__':
    app = QApplication(sys.argv)  # 在 QApplication 方法中使用，创建应用程序对象
    # app.setWindowIcon(QIcon("Images/Earth.ico"))# 为程序设置图标
    myWin = MyMainWindow()  # 实例化 MyMainWindow 类，创建主窗口
    myWin.setWindowIcon(QIcon("Images/Earth.ico"))
    myWin.show()  # 在桌面显示控件 myWin
    sys.exit(app.exec_())

