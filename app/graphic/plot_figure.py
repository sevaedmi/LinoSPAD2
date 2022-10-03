import sys
from PyQt5.QtWidgets import QWidget, QDialog, QApplication, QPushButton,\
    QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg\
    as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT\
    as NavigationToolbar
import matplotlib.pyplot as plt
import random


class PltCanvas(QWidget):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super(PltCanvas, self).__init__(parent)
        # a figure instance to plot on
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        # creating a Vertical Box layout
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

        self.toolbar = NavigationToolbar(self.canvas, self)
        # adding tool bar to the layout
        self.layout.addWidget(self.toolbar)

        # random data
        data = [random.random() for i in range(10)]
        # clearing old figure
        self.figure.clear()
        # create an axis
        ax = self.figure.add_subplot(111)

        # plot data
        ax.plot(data, '*-')

        # refresh canvas
        self.canvas.draw()
