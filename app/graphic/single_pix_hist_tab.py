from PyQt5.QtWidgets import QWidget
from app.graphic.ui.SinglePixelHistogram_ui import Ui_SinglePixelHistogram


class SinglePixelHistogram(QWidget, Ui_SinglePixelHistogram):
    def __init__(self, parent=None):

        super().__init__(parent)

        self.setupUi(self)
