# from PyQt5.QtWidgets import QWidget
# from app.graphic.ui.SinglePixelHistogram_ui import Ui_SinglePixelHistogram
from PyQt5 import QtWidgets, uic

# class SinglePixelHistogram(QWidget, Ui_SinglePixelHistogram):
#     def __init__(self, parent=None):

#         super().__init__(parent)

#         self.setupUi(self)


class SinglePixelHistogram(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        uic.loadUi("app/graphic/ui/SinglePixelHistogram_tab_c.ui", self)
        self.show()
