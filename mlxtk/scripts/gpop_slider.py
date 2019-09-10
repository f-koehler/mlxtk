import argparse
import sys
from pathlib import Path

from PySide2 import QtWidgets

from .. import inout, plot, units
from ..tools.gpop import transform_to_momentum_space
from ..ui import MatplotlibWidget, load_ui


class GpopSlider(QtWidgets.QWidget):
    def __init__(self,
                 times,
                 grids,
                 gpops,
                 plot_args,
                 momentum_space: bool = False,
                 parent=None):
        super().__init__(parent)

        self.times = times
        self.grids = grids
        self.gpops = gpops
        self.plot_args = plot_args
        self.time_index = 0
        self.dof = 1
        self.line = None  # type: Line2D
        self.momentum_space = momentum_space

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
        unit_system = units.get_default_unit_system()

        if self.line:
            self.line.remove()
            self.line = None

        self.line = self.axes.plot(self.grids[self.dof],
                                   self.gpops[self.dof][self.time_index])[0]
        if self.momentum_space:
            # self.axes.set_xlabel(unit_system.get_momentum_label().format_label("p"))
            self.axes.set_ylabel(r"$\rho_1(k,t)$")
        else:
            self.axes.set_xlabel(
                unit_system.get_length_unit().format_label("x"))
            self.axes.set_ylabel(r"$\rho_1(x,t)$")
        plot.apply_2d_args(self.axes, self.plot_args)
        self.axes.set_ylim([
            self.axes.get_ylim()[0],
            1.02 * self.gpops[self.dof][self.time_index].max()
        ])
        self.plot.canvas.draw()

    def update_plot(self, index: int):
        del index

        self.line.set_ydata(self.gpops[self.dof][self.time_index])
        self.axes.set_ylim([
            self.axes.get_ylim()[0],
            1.02 * self.gpops[self.dof][self.time_index].max()
        ])
        self.plot.canvas.draw()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path",
                        type=Path,
                        default=Path("propagate.h5/gpop"),
                        help="path to the gpop file")
    parser.add_argument("--momentum",
                        action="store_true",
                        default=False,
                        help="whether to transform to momentum space")
    plot.add_argparse_2d_args(parser)
    args = parser.parse_args()

    data = inout.read_gpop(args.path)
    if args.momentum:
        data = transform_to_momentum_space(data)

    app = QtWidgets.QApplication(sys.argv)
    form = GpopSlider(*data, args, momentum_space=args.momentum)
    assert form
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
