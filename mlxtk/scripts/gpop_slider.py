import argparse
import os
import sys

import matplotlib
import pkg_resources
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PySide2 import QtCore, QtUiTools, QtWidgets

from .. import inout
from .. import units
from .. import plot


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


class GpopSlider(QtWidgets.QWidget):
    def __init__(self, times, grids, gpops, plot_args, parent=None):
        super().__init__(parent)

        self.times = times
        self.grids = grids
        self.gpops = gpops
        self.plot_args = plot_args
        self.time_index = 0
        self.dof = 1
        self.line = None  # type: Line2D

        ui_file = QtCore.QFile(
            pkg_resources.resource_filename("mlxtk.ui", "gpop_slider.ui"))
        ui_file.open(QtCore.QFile.ReadOnly)
        loader = QtUiTools.QUiLoader()
        loader.registerCustomWidget(MatplotlibWidget)
        self.window = loader.load(ui_file, self)
        ui_file.close()

        self.plot = self.window.findChild(MatplotlibWidget,
                                          "plot")  # type: MatplotlibWidget
        self.slider_time = self.window.findChild(
            QtWidgets.QSlider, "slider_time")  # type: QtWidgets.QSlider
        self.spin_time = self.window.findChild(
            QtWidgets.QSpinBox, "spin_time")  # type: QtWidgets.QSpinBox
        self.label_time = self.window.findChild(
            QtWidgets.QLabel, "label_time")  # type: QtWidgets.QLabel
        self.spin_dof = self.window.findChild(
            QtWidgets.QSpinBox, "spin_dof")  # type: QtWidgets.QSpinBox
        self.dof_control = self.window.findChild(
            QtWidgets.QWidget, "dof_control")  # type: QtWidgets.QWidget

        self.axes = self.plot.figure.subplots(1,1)
        self.plot.figure.set_tight_layout(True)

        self.slider_time.setTracking(True)
        self.slider_time.setMaximum(len(self.times) - 1)
        self.slider_time.valueChanged.connect(self.spin_time.setValue)

        self.spin_time.setMaximum(len(self.times) - 1)
        self.spin_time.valueChanged.connect(self.slider_time.setValue)
        self.spin_time.valueChanged.connect(self.set_time)
        self.spin_time.valueChanged.connect(self.update_plot)

        self.spin_dof.setMaximum(len(self.grids))
        if len(self.grids) < 2:
            self.dof_control.hide()

        self.set_time(self.time_index)
        self.init_plot()

        self.window.show()

    def set_time(self, index: int):
        self.time_index = index
        self.label_time.setText("Time: {:.4E}".format(self.times[index]))

    def init_plot(self):
        if self.line:
            self.line.remove()
            self.line = None

        self.line = self.axes.plot(self.grids[self.dof], self.gpops[self.dof][self.time_index])[0]
        self.axes.set_xlabel(units.get_length_label())
        self.axes.set_ylabel(r"$\rho_1(x,t)$")
        plot.apply_2d_args(self.axes, self.plot_args)
        self.axes.set_ylim([self.axes.get_ylim()[0], 1.02 * self.gpops[self.dof][self.time_index].max()])
        self.plot.canvas.draw()

    def update_plot(self, index: int):
        self.line.set_ydata(self.gpops[self.dof][self.time_index])
        self.axes.set_ylim([self.axes.get_ylim()[0], 1.02 * self.gpops[self.dof][self.time_index].max()])
        self.plot.canvas.draw()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to the gpop file")
    plot.add_argparse_2d_args(parser)
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    form = GpopSlider(*inout.read_gpop(args.path), args)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
