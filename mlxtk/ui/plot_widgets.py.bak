import PyQt5.QtWidgets
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.axes = self.fig.add_subplot(1, 1, 1)

        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)

        super(MplCanvas, self).setSizePolicy(
            PyQt5.QtWidgets.QSizePolicy.Expanding,
            PyQt5.QtWidgets.QSizePolicy.Expanding)
        super(MplCanvas, self).updateGeometry()


class SingleLinePlot(MplCanvas):
    def __init__(self, x, y, parent=None, **kwargs):
        super(SingleLinePlot, self).__init__(parent)

        self.line, = self.axes.plot(
            x,
            y,
            linestyle=kwargs.get("linestyle", "-"),
            marker=kwargs.get("marker", "."),
        )

        if "xlabel" in kwargs:
            self.axes.set_xlabel(kwargs["xlabel"])

        if "ylabel" in kwargs:
            self.axes.set_ylabel(kwargs["ylabel"])
