from PyQt5.QtWidgets import QPushButton, QWidget, QTabWidget, QVBoxLayout,\
    QFileDialog, QLineEdit
from PyQt5.QtCore import pyqtSlot
from functions import plot_valid
from app.graphic.plot_figure import PltCanvas

import matplotlib
matplotlib.use('Qt5Agg')


class TableWidget(QWidget):

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tabs.resize(300, 200)

        # Add tabs
        self.tabs.addTab(self.tab1, "Tab 1")
        self.tabs.addTab(self.tab2, "Tab 2")

        # Create first tab
        self.tab1.layout = QVBoxLayout(self)
        self.pushButtonLoadPath = QPushButton("Set path")
        self.lineEditPath = QLineEdit('')
        self.pushButtonStartSync = QPushButton("Show hits from LinoSPAD2")
        self.tab1.layout.addWidget(self.pushButtonLoadPath)
        self.tab1.layout.addWidget(self.lineEditPath)
        self.tab1.layout.addWidget(self.pushButtonStartSync)

        self.plotWidget = PltCanvas(self)
        self.tab1.layout.addWidget(self.plotWidget)

        self.tab1.setLayout(self.tab1.layout)

        # adding tool bar to the layout

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)
        self.pushButtonLoadPath.clicked.connect(self.load_path)
        self.pushButtonStartSync.clicked.connect(self.start_stream)

    @pyqtSlot()
    def load_path(self):
        file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.lineEditPath.setText(file)
        print(file + "\n")

    def start_stream(self):
        plot_valid.online_plot_valid(self.lineEditPath.text(), range(256), 512)
