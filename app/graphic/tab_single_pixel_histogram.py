""" Script for calculating a histogram of a single pixel

"""

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QFormLayout,
    QSizePolicy,
)
from PyQt5.QtCore import QSize
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import pyplot as plt
import os
import glob
import numpy as np
from app.tools import unpack_data as f_up

# TODO: fix widget size, position, relative positioning, size-handling when the main
# window size is changed


class SinglePixelHistogram(QWidget):
    def __init__(self, height=4, width=4, dpi=100):
        super().__init__()

        # figure initialization
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # buttons
        self.folder = ""
        self.buttonbrowse = QPushButton("Browse")
        self.buttonbrowse.setMinimumSize(QSize(100, 35))
        self.buttonplot = QPushButton("Plot histogram")
        self.buttonbrowse.clicked.connect(self._get_dir)
        self.buttonplot.clicked.connect(self._plot_hist)

        # text fields
        self.path = QLineEdit()
        self.path.setMinimumSize(QSize(400, 35))

        # layout
        outerLayout = QVBoxLayout()
        topLayout = QFormLayout()
        bottomLayout = QFormLayout()

        topLayout.addRow(self.buttonbrowse, self.path)

        bottomLayout.addWidget(self.buttonbrowse)
        bottomLayout.addWidget(self.path)
        bottomLayout.addWidget(self.buttonplot)
        bottomLayout.addWidget(self.toolbar)
        bottomLayout.addWidget(self.canvas)

        outerLayout.addLayout(topLayout)
        outerLayout.addLayout(bottomLayout)
        self.setLayout(outerLayout)

    # TODO: add dialog window for inputting pixel number
    def _plot_hist(self):

        pix = 150

        os.chdir(self.folder)

        file = glob.glob("*.dat*")[0]

        data = f_up.unpack_binary_flex(file, 512)

        bins = np.arange(0, 4e9, 17.867 * 1e6)  # bin size of 17.867 us
        plt.ioff()
        histogram = plt.hist(data[pix], bins=bins, color="teal")[0]

        self.axes.plot(histogram)
        self.axes.set_xlabel("Time [ps]")
        self.axes.set_ylabel("# of timestamps [-]")
        self.axes.set_title("Pixel {} histogram".format(pix))
        self.canvas.draw()
        self.canvas.flush_events()

        try:
            os.chdir("results/single pixel histograms")
        except Exception:
            os.mkdir("results/single pixel histograms")
            os.chdir("results/single pixel histograms")
        plt.savefig("{file}, pixel {pixel}.png".format(file=file, pixel=pix))
        os.chdir("../..")

    def _get_dir(self):
        self.folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.path.setText(self.folder)
