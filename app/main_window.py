from PyQt5.QtWidgets import QMainWindow, QApplication
from app.graphic.ui.mainwindow_ui import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):

        super().__init__(parent)

        self.setupUi(self)

    # for stopping the script upon closing the app window
    def closeEvent(self, event):
        QApplication.quit()
