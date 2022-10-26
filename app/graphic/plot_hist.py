""" Class for a histogram plotting function for the LinoSPAD2 data analysis
and visualization application.

"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLineEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import random
import matplotlib

matplotlib.use("Qt5Agg")


class PlotHistogram(QWidget):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        Initialization of the class. A widget for the plot together
        with a toolbar will be initialized.

        Parameters
        ----------
        parent : basestring, optional
            Class being inherited from. The default is None.
        width : float, optional
            Widget width. The default is 5.
        height : float, optional
            Widget height. The default is 4.
        dpi : int, optional
            Scale of the widget. The default is 100.

        Returns
        -------
        None.

        """
        super(PlotHistogram, self).__init__(parent)

        # a figure instance to plot on
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvas(self.figure)
        # creating a Vertical Box layout
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

        self.toolbar = NavigationToolbar(self.canvas, self)
        # adding tool bar to the layout
        self.layout.addWidget(self.toolbar)

        # random data
        data = [random.random() for i in range(300)]
        # clearing old figure
        self.figure.clear()
        # create an axis
        ax = self.figure.add_subplot(111)

        # plot data
        ax.hist(data, bins=30)
        ax.set_xlabel("\u0394t [ps]")
        ax.set_ylabel("Timestamps [-]")

        # refresh canvas
        self.canvas.draw()
