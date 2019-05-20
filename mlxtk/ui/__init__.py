import os

import pkg_resources
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PySide2 import QtCore, QtUiTools, QtWidgets

if os.environ["QT_API"] != "pyside2":
    raise RuntimeError(
        "Please set the QT_API environment variable to \"pyside2\" for now.")


class MatplotlibWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.navbar = NavigationToolbar(self.canvas, self)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.navbar)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)


def load_ui(name: str, parent: QtWidgets.QWidget = None) -> QtWidgets.QWidget:
    ui_file = QtCore.QFile(pkg_resources.resource_filename("mlxtk.ui", name))
    ui_file.open(QtCore.QFile.ReadOnly)
    loader = QtUiTools.QUiLoader()
    loader.registerCustomWidget(MatplotlibWidget)
    window = loader.load(ui_file, parent)
    ui_file.close()
    return window
