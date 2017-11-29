import argparse
import matplotlib
import sys

from mlxtk.plot.plot import Plot

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from mlxtk.plot.qt5_plot import Qt5Plot
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    pass


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, title, **kwargs):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle(title)

        self.main_widget = QtWidgets.QWidget()

        self.plot = Qt5Plot(
            self.main_widget, projection=kwargs.get("projection", None))

        self.plot.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.plot.setFocus()
        self.setCentralWidget(self.plot)
        self.plot.figure.tight_layout()
        self.plot.updateGeometry()

        self.toolbar = NavigationToolbar(self.plot, self.main_widget)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.plot)
        self.main_widget.setLayout(self.layout)
        self.setCentralWidget(self.main_widget)


class SimplePlotProgram(object):
    def __init__(self, title, init_plot, update_plot=None, **plot_args):
        self.title = title
        self.init_plot = init_plot
        self.update_plot = update_plot
        self.plot_args = plot_args

    def main(self, args):
        if args.output_file is None:
            matplotlib.use("Qt5Agg")

            application = QtWidgets.QApplication(sys.argv)
            window = ApplicationWindow(self.title, **self.plot_args)
            self.init_plot(window.plot)
            apply_plot_parameters(window.plot, args)
            window.show()
            sys.exit(application.exec_())
        else:
            plot = Plot(**self.plot_args)
            self.init_plot(plot)
            apply_plot_parameters(plot, args)
            plot.tight_layout()
            plot.figure.savefig(args.output_file)


def apply_plot_parameters(plot, args):
    if args.logx:
        plot.axes.set_xscale("log")

    if args.logy:
        plot.axes.set_yscale("log")

    plot.axes.grid(not args.no_grid)

    if args.xmin is not None:
        _, xmax = plot.axes.get_xlim()
        plot.axes.set_xlim(args.xmin, xmax)

    if args.xmax is not None:
        xmin, _ = plot.axes.get_xlim()
        plot.axes.set_xlim(xmin, args.xmax)

    if args.ymin is not None:
        _, ymax = plot.axes.get_ylim()
        plot.axes.set_ylim(args.ymin, ymax)

    if args.ymax is not None:
        ymin, _ = plot.axes.get_ylim()
        plot.axes.set_ylim(ymin, args.ymax)
