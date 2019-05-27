import argparse
import sys

from PySide2 import QtWidgets

from .. import inout, plot, units
from ..ui import MatplotlibWidget, load_ui


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

        self.window = load_ui("gpop_slider.ui")

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

        self.axes = self.plot.figure.subplots(1, 1)
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

        self.line = self.axes.plot(self.grids[self.dof],
                                   self.gpops[self.dof][self.time_index])[0]
        self.axes.set_xlabel(units.get_length_label())
        self.axes.set_ylabel(r"$\rho_1(x,t)$")
        plot.apply_2d_args(self.axes, self.plot_args)
        self.axes.set_ylim([
            self.axes.get_ylim()[0],
            1.02 * self.gpops[self.dof][self.time_index].max()
        ])
        self.plot.canvas.draw()

    def update_plot(self, index: int):
        self.line.set_ydata(self.gpops[self.dof][self.time_index])
        self.axes.set_ylim([
            self.axes.get_ylim()[0],
            1.02 * self.gpops[self.dof][self.time_index].max()
        ])
        self.plot.canvas.draw()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to the gpop file")
    plot.add_argparse_2d_args(parser)
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    form = GpopSlider(*inout.read_gpop(args.path), args)
    assert form
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
