import matplotlib.figure
from PyQt5 import QtWidgets
from matplotlib.backends import backend_qt5agg as backend
import numpy


class Qt5Plot(backend.FigureCanvas):
    def __init__(self, parent):
        self.figure = matplotlib.figure.Figure()
        self.axes = self.figure.add_subplot(1, 1, 1)

        backend.FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)
        backend.FigureCanvas.setSizePolicy(self,
                                           QtWidgets.QSizePolicy.Expanding,
                                           QtWidgets.QSizePolicy.Expanding)
        backend.FigureCanvas.updateGeometry(self)
