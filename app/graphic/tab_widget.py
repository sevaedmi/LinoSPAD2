from PyQt5.QtWidgets import QPushButton, QWidget, QTabWidget,QVBoxLayout, QFileDialog, QLineEdit
from PyQt5.QtCore import pyqtSlot
from functions import plot_valid
from app.graphic.plot_figure import PltCanvas
from app.graphic.tab_livehistogram import LiveHistogram
import matplotlib
matplotlib.use('Qt5Agg')


class TableWidget(QWidget):

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = LiveHistogram(self)
        self.tab2 = QWidget()
        self.tabs.resize(300, 200)

        # Add tabs
        self.tabs.addTab(self.tab1, "Live histogram")
        self.tabs.addTab(self.tab2, "Tab 2")

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

