from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5_agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure()
        self.axes = fig.add_subplot(111)

        self.init_figure()

        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def init_figure(self):
        pass
