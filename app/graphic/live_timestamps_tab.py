from PyQt5.QtWidgets import QWidget
from app.ui.LiveTimestamps_tab_ui import Ui_LiveTimestamps


class LiveTimestamps(QWidget, Ui_LiveTimestamps):
    def __init__(self, parent=None):

        super().__init__(parent)

        self.setupUi(self)
