"""
进度条百分比
"""

'''
    params：
    1、百分比 percentage
    2、显示的文字 text
'''
import time
from PyQt5.QtCore import QThread, pyqtSignal


class CalculatorThread(QThread):
    signal_progress_update = pyqtSignal(list)

    def __init__(self):
        super(CalculatorThread, self).__init__()
        self.total = 100
        self.i = 0
    def run(self):
        while True:
            self.i += 1
            self.signal_progress_update.emit([self.i, self.total])
            time.sleep(0.02)
            if self.i > self.total:
                break