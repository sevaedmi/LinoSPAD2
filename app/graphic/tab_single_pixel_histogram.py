""" Widget for the plot of a histogram for a single pixel for the LinoSPAD2
data analysis and visualization application.

"""

from PyQt5.QtWidgets import (
    QPushButton,
    QWidget,
    QTabWidget,
    QVBoxLayout,
    QFileDialog,
    QLineEdit,
)
from PyQt5.QtCore import pyqtSlot
from functions import plot_valid
from app.graphic.plot_hist import PlotHistogram

import matplotlib

matplotlib.use("Qt5Agg")


class SinglePixelHistogram(QWidget):
    def __init__(self, parent):
        """ Initialization of the class. Widget with an input window for path
        to a data file and a histogram widget 'PlotHistogram' will be
        initialized.

        Parameters
        ----------
        parent : basestring
            Class being inherited from.

        Returns
        -------
        None.

        """
        super(SinglePixelHistogram, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        # Create first tab
        self.pushButtonLoadPath = QPushButton("Set path")
        self.lineEditPath = QLineEdit("")
        self.pushButtonStartSync = QPushButton("Show hits from LinoSpad")
        self.plotWidget = PlotHistogram(self)

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
