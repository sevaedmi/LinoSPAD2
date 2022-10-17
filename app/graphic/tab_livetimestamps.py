import glob
import os

from PyQt5.QtWidgets import QPushButton, QWidget, QTabWidget, QVBoxLayout, QFileDialog, QLineEdit
from PyQt5.QtCore import pyqtSlot, QTimer

from app.tools import timestamp_computation
from functions import plot_valid
from app.graphic.plot_figure import PltCanvas
import app.tools.timestamp_computation
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')


class LiveTimestamps(QWidget):

    def __init__(self, parent):
        super(LiveTimestamps, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        # Create first tab
        self.pushButtonLoadPath = QPushButton("Set path")
        self.lineEditPath = QLineEdit('')
        self.pushButtonStartSync = QPushButton("Start stream")
        self.plotWidget = PltCanvas(self)
        self.timer = QTimer()
        self.timerRunning = False
        self.last_file_ctime = 0
        self.pathtotimestamp = ''
        self.layout.addWidget(self.pushButtonLoadPath)
        self.layout.addWidget(self.lineEditPath)
        self.layout.addWidget(self.pushButtonStartSync)
        self.layout.addWidget(self.plotWidget)

        self.setLayout(self.layout)

        # Add tabs to widget
        self.pushButtonLoadPath.clicked.connect(self.slot_loadpath)
        self.pushButtonStartSync.clicked.connect(self.slot_startstream)
        self.timer.timeout.connect(self.updateTimeStamp)

    def updateTimeStamp(self):
        DATA_FILES = glob.glob('*.dat*')
        os.chdir(self.pathtotimestamp)
        print("updateTimeStamp: timer running")
        try:
            last_file = max(DATA_FILES, key=os.path.getctime)
            new_file_ctime = os.path.getctime(last_file)

            if new_file_ctime > self.last_file_ctime:
                # print("updateTimeStamp: new file")
                self.last_file_ctime = new_file_ctime
                # print("updateTimeStamp:" + self.pathtotimestamp + last_file)
                validtimestemps, peak = timestamp_computation.get_nmr_validtimestamps(self.pathtotimestamp + '/' + last_file, np.arange(145, 155, 1), 512)

                self.plotWidget.setPlotData(np.arange(0, 256, 1),validtimestemps,peak)
            # else:
                # print("updateTimeStamp:  not a new file")

        except ValueError:
             print("updateTimeStamp:  waiting for a file")


    @pyqtSlot()
    def slot_loadpath(self):
        file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.lineEditPath.setText(file)
        self.pathtotimestamp = file
        print(file + "\n")

    def slot_startstream(self):
        self.last_file_ctime = 0

        if self.timerRunning is True:
            self.timer.stop()
            self.timerRunning = False
            self.pushButtonStartSync.setText('Start stream')
        else:
            self.pushButtonStartSync.setText('Stop stream')
            self.timer.start(100)
            self.timerRunning = True
