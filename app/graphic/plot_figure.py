import sys
from PyQt5.QtWidgets import QWidget, QDialog, QApplication, QPushButton,\
    QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg\
    as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT \
    as NavigationToolbar
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import AutoMinorLocator


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
        plt.rcParams.update({"font.size": 18})
        # random data
        data =[1 for i in range(256)]
        # clearing old figure
        self.figure.clear()
        # create an axis
        self.ax = self.figure.add_subplot(111)

        # plot data
        self.plot, = self.ax.plot(data, '-ok')
        self.setplotparameters()
        # refresh canvas
        self.canvas.draw()

    def setplotion(self):
        self.figure.ion()

    def setplotparameters(self):
        plt.rcParams.update({"font.size": 18})
        plt.xlabel("Pixel [-]")
        plt.ylabel("Valid timestamps [-]")

        plt.box(bool(1))
        plt.grid(False)
        plt.subplots_adjust(left=0.15)

        self.ax.tick_params(which='both', width=2, direction="in")
        self.ax.tick_params(which='major', length=7, direction="in")
        self.ax.tick_params(which='minor', length=4, direction="in")
        self.ax.yaxis.set_ticks_position('both')
        self.ax.xaxis.set_ticks_position('both')

        for axis in ['top', 'bottom', 'left', 'right']:
            self.ax.spines[axis].set_linewidth(2)

    def setPlotData(self, xdataplot, yplotdata, peak):

        # self.plot.set_xdata(xdataplot)

        self.plot.set_ydata(yplotdata)
        # self.ax.set_title("Maximum counts is {}".format(peak))
        self.ax.relim()
        self.ax.autoscale_view()
        self.setplotparameters()

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def setPlotScale(self, scaleLin=True):
        if scaleLin:
            self.ax.set_yscale('linear')
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
        else:
            self.ax.set_yscale('log')
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()




