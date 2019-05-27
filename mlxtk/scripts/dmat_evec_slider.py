import argparse
import sys

import numpy
from PySide2 import QtWidgets

from .. import inout, plot, units
from ..ui import MatplotlibWidget, load_ui


class DmatEvecSlider(QtWidgets.QWidget):
    def __init__(self, times, grid, evecs, plot_args, parent=None):
        super().__init__(parent)

        self.times = times
        self.grid = grid
        self.evecs = evecs
        self.plot_args = plot_args
        self.time_index = 0
        self.index = 0
        self.line_real = None  # type: Line2D
        self.line_imag = None  # type: Line2D
        self.line_abs = None  # type: Line2D

        self.window = load_ui(
            "dmat_evec_slider.ui")  # type: QtWidgets.QMainWindow

        self.plot = self.window.findChild(MatplotlibWidget,
                                          "plot")  # type: MatplotlibWidget
        self.slider_time = self.window.findChild(
            QtWidgets.QSlider, "slider_time")  # type: QtWidgets.QSlider
        self.spin_time = self.window.findChild(
            QtWidgets.QSpinBox, "spin_time")  # type: QtWidgets.QSpinBox
        self.label_time = self.window.findChild(
            QtWidgets.QLabel, "label_time")  # type: QtWidgets.QLabel
        self.spin_index = self.window.findChild(
            QtWidgets.QSpinBox, "spin_index")  # type: QtWidgets.QSpinBox
        self.check_real = self.window.findChild(
            QtWidgets.QCheckBox, "check_real")  # type: QtWidgets.QCheckBox
        self.check_imag = self.window.findChild(
            QtWidgets.QCheckBox, "check_imag")  # type: QtWidgets.QCheckBox
        self.check_abs = self.window.findChild(
            QtWidgets.QCheckBox, "check_absolute")  # type: QtWidgets.QCheckBox

        self.axes = self.plot.figure.subplots(1, 1)
        self.plot.figure.set_tight_layout(True)

        self.slider_time.setTracking(True)
        self.slider_time.setMaximum(len(self.times) - 1)
        self.slider_time.valueChanged.connect(self.spin_time.setValue)

        self.spin_time.setMaximum(len(self.times) - 1)
        self.spin_time.valueChanged.connect(self.slider_time.setValue)
        self.spin_time.valueChanged.connect(self.set_time)
        self.spin_time.valueChanged.connect(self.set_time_index)

        self.check_real.toggled.connect(self.toggle_real)
        self.check_imag.toggled.connect(self.toggle_imag)
        self.check_abs.toggled.connect(self.toggle_abs)

        self.spin_index.setMaximum(self.evecs.shape[0] - 1)
        self.spin_index.valueChanged.connect(self.set_index)

        self.set_time(self.time_index)
        self.init_plot()

        self.window.show()

    def set_time(self, index: int):
        self.time_index = index
        self.label_time.setText("Time: {:.4E}".format(self.times[index]))

    def init_plot(self):
        if self.line_real:
            self.line_real.remove()
            self.line_real = None

        if self.line_imag:
            self.line_imag.remove()
            self.line_imag = None

        if self.line_abs:
            self.line_abs.remove()
            self.line_abs = None

        self.line_abs = self.axes.plot(
            self.grid, numpy.abs(self.evecs[self.index, self.time_index]))[0]
        self.line_real = self.axes.plot(
            self.grid, numpy.real(self.evecs[self.index, self.time_index]))[0]
        self.line_imag = self.axes.plot(
            self.grid, numpy.imag(self.evecs[self.index, self.time_index]))[0]

        self.line_abs.set_visible(self.check_abs.isChecked())
        self.line_real.set_visible(self.check_real.isChecked())
        self.line_imag.set_visible(self.check_imag.isChecked())

        self.axes.set_xlabel(units.get_length_label())
        self.axes.set_ylabel(r"$\varphi(x)$")
        plot.apply_2d_args(self.axes, self.plot_args)
        self.update_yrange()

        self.plot.canvas.draw()

    def toggle_real(self, enabled: bool):
        self.line_real.set_visible(enabled)
        self.update_yrange()
        self.plot.canvas.draw()

    def toggle_imag(self, enabled: bool):
        self.line_imag.set_visible(enabled)
        self.update_yrange()
        self.plot.canvas.draw()

    def toggle_abs(self, enabled: bool):
        self.line_abs.set_visible(enabled)
        self.update_yrange()
        self.plot.canvas.draw()

    def set_index(self, index: int):
        self.index = index

        self.line_real.set_ydata(
            numpy.real(self.evecs[self.index, self.time_index]))
        self.line_imag.set_ydata(
            numpy.imag(self.evecs[self.index, self.time_index]))
        self.line_abs.set_ydata(
            numpy.abs(self.evecs[self.index, self.time_index]))

        self.update_yrange()

        self.plot.canvas.draw()

    def set_time_index(self, index: int):
        self.line_real.set_ydata(numpy.real(self.evecs[self.index, index]))
        self.line_imag.set_ydata(numpy.imag(self.evecs[self.index, index]))
        self.line_abs.set_ydata(numpy.abs(self.evecs[self.index, index]))

        self.update_yrange()

        self.plot.canvas.draw()

    def update_yrange(self):
        y_min, y_max = self.axes.get_ylim()

        def safety_min(x: float) -> float:
            if x < 0.:
                return x * 1.02
            return 0.98 * x

        def safety_max(x: float) -> float:
            if x < 0.:
                return x * 0.98
            return 1.02 * x

        if self.check_real.isChecked():
            y_min = min(y_min, safety_min(self.line_real.get_ydata().min()))
            y_max = max(y_min, safety_max(self.line_real.get_ydata().max()))

        if self.check_imag.isChecked():
            y_min = min(y_min, safety_min(self.line_imag.get_ydata().min()))
            y_max = max(y_min, safety_max(self.line_imag.get_ydata().max()))

        if self.check_abs.isChecked():
            y_min = min(y_min, safety_min(self.line_abs.get_ydata().min()))
            y_max = max(y_min, safety_max(self.line_abs.get_ydata().max()))

        self.axes.set_ylim(y_min, y_max)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="evec_dmat_dof1_spf",
        help="path to the file containing the eigenvectors of the dmat")
    plot.add_argparse_2d_args(parser)
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    window = DmatEvecSlider(*inout.read_dmat_evecs_grid(args.path), args)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
