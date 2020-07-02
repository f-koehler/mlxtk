import argparse
import sys
from typing import Optional, Tuple

import matplotlib
import numpy
from PySide2 import QtWidgets

from mlxtk import plot, units
from mlxtk.inout.gpop import read_gpop
from mlxtk.plot.gpop import plot_gpop
from mlxtk.tools.gpop import transform_to_momentum_space
from mlxtk.ui.plot_slider import PlotSlider


class Gui(PlotSlider):
    def __init__(
        self,
        data: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray],
        min_index: int,
        max_index: int,
        projection: str = None,
        plot_args: Optional[argparse.Namespace] = None,
        parent: QtWidgets.QWidget = None,
    ):
        self.time, self.grid, self.density = data
        self.line: matplotlib.lines.Line2D = None
        super().__init__(min_index, max_index, plot_args=plot_args, parent=parent)

        self.spin.valueChanged.connect(self.update_label)

    def init_plot(self):
        self.label.setText("Time: {:6.2f}".format(self.time[0]))

        if self.line:
            self.line.remove()
            self.line = None

        unit_system = units.get_default_unit_system()
        self.axes.set_xlabel(unit_system.get_length_unit().format_label("x_1"))
        self.axes.set_ylabel(unit_system.get_time_unit().format_label("t"))

        (self.line,) = self.axes.plot(self.grid, self.density[0])

        if self.plot_args:
            plot.apply_2d_args(self.axes, self.plot.figure, self.plot_args)

        self.plot.canvas.draw()

    def update_plot(self, index: int):
        self.line.set_ydata(self.density[index])
        self.plot.canvas.draw()

    def update_label(self, index: int):
        self.label.setText("Time: {:6.2f}".format(self.time[index]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", nargs="?", default="propagate.h5/gpop", help="path to the gpop file"
    )
    parser.add_argument("-d", "--dof", type=int, default=1, help="degree of freedom")
    parser.add_argument(
        "--momentum", action="store_true", help="transform to momentum space"
    )
    plot.add_argparse_2d_args(parser)
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    data = read_gpop(args.path, args.dof)
    if args.momentum:
        data = transform_to_momentum_space(data)
    gui = Gui(data, 0, len(data[0]) - 1, plot_args=args)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
