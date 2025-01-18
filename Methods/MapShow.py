"""
展示图形可视化
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import pyplot

pyplot.rcParams['font.sans-serif'] = ['SimHei']
pyplot.rcParams['axes.unicode_minus'] = False





class MyFigure(FigureCanvasQTAgg):
    def __init__(self, width=5, height=4, dpi=200):
        # 1、创建一个绘制窗口Figure对象
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # 2、在父类中激活Figure窗口,同时继承父类属性
        super(MyFigure, self).__init__(self.fig)

    # 这里就是绘制图像、示例
    def plotSin(self, x, y):
        self.axes0 = self.fig.add_subplot(111)
        self.axes0.plot(x, y)
