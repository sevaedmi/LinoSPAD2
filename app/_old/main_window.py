from PyQt5.QtWidgets import QMainWindow, QApplication
from app.graphic.tab_widget import TableWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # self.setWindowTitle("My App")
        # button = QPushButton("Press Me!")
        # Set the central widget of the Window.
        # self.setCentralWidget(button)
        # self.setFixedSize(QSize(400, 300))
        self.table_widget = TableWidget(self)
        self.setCentralWidget(self.table_widget)

    def closeEvent(self, event):
        QApplication.quit()
