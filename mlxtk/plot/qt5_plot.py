from PyQt5 import QtWidgets
from matplotlib.backends import backend_qt5agg as backend
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.figure
import numpy


class Qt5Plot(backend.FigureCanvas):
    def __init__(self, parent, projection=None):
        self.figure = matplotlib.figure.Figure()
        backend.FigureCanvas.__init__(self, self.figure)

        if projection is None:
            self.axes = self.figure.add_subplot(1, 1, 1)
        else:
            self.axes = self.figure.add_subplot(1, 1, 1, projection=projection)

        self.setParent(parent)
        backend.FigureCanvas.setSizePolicy(self,
                                           QtWidgets.QSizePolicy.Expanding,
                                           QtWidgets.QSizePolicy.Expanding)
        backend.FigureCanvas.updateGeometry(self)
