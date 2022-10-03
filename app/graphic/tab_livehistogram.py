from PyQt5.QtWidgets import QPushButton, QWidget, QTabWidget, QVBoxLayout,\
    QFileDialog, QLineEdit
from PyQt5.QtCore import pyqtSlot
from functions import plot_valid
from app.graphic.plot_figure import PltCanvas

import matplotlib

matplotlib.use('Qt5Agg')


class LiveHistogram(QWidget):

    def __init__(self, parent):
        super(LiveHistogram, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        # Create first tab
        self.pushButtonLoadPath = QPushButton("Set path")
        self.lineEditPath = QLineEdit('')
        self.pushButtonStartSync = QPushButton("Show hits from LinoSpad")
        self.plotWidget = PltCanvas(self)

        self.layout.addWidget(self.pushButtonLoadPath)
        self.layout.addWidget(self.lineEditPath)
        self.layout.addWidget(self.pushButtonStartSync)
        self.layout.addWidget(self.plotWidget)

        self.setLayout(self.layout)

        # Add tabs to widget
        self.pushButtonLoadPath.clicked.connect(self.load_path)
        self.pushButtonStartSync.clicked.connect(self.start_stream)

    @pyqtSlot()
    def load_path(self):
        file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.lineEditPath.setText(file)
        print(file + "\n")

    def start_stream(self):
        plot_valid.online_plot_valid(self.lineEditPath.text(), range(256), 512)
