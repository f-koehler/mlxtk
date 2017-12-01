#!/usr/bin/env python
import argparse
import matplotlib
import sys

import mlxtk.plot.argparser
import mlxtk.plot.plot_program
from mlxtk.plot.plot_program import QtWidgets, QtCore, apply_plot_parameters
from mlxtk.inout.gpop import read_gpop

density = None
grid = None
times = None


class ApplicationWindow(mlxtk.plot.plot_program.ApplicationWindow):
    def __init__(self, title, **kwargs):
        mlxtk.plot.plot_program.ApplicationWindow.__init__(
            self, title, **kwargs)

        self.slider_widget = QtWidgets.QWidget(self.main_widget)
        self.layout.addWidget(self.slider_widget)

        self.slider_layout = QtWidgets.QHBoxLayout()
        self.slider_widget.setLayout(self.slider_layout)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal,
                                        self.slider_widget)
        self.slider_layout.addWidget(self.slider)

        self.slider_label = QtWidgets.QLabel("time: ")
        self.slider_layout.addWidget(self.slider_label)

        self.slider.valueChanged.connect(self.update_label)

    def update_label(self, index):
        global times
        self.slider_label.setText("time: {}".format(str(index)))


class SimplePlotProgram(object):
    def __init__(self, title, init_plot, update_plot=None, **plot_args):
        self.title = title
        self.init_plot = init_plot
        self.update_plot = update_plot
        self.plot_args = plot_args

    def main(self, args):
        matplotlib.use("Qt5Agg")

        application = QtWidgets.QApplication(sys.argv)
        window = ApplicationWindow(self.title, **self.plot_args)
        window.slider.setMaximum(len(times) - 1)
        self.init_plot(window.plot)

        def call_update_plot(index):
            self.update_plot(index)
            window.plot.figure.canvas.draw_idle()

        window.slider.valueChanged.connect(call_update_plot)
        apply_plot_parameters(window.plot, args)
        window.show()
        sys.exit(application.exec_())


def main():
    parser = argparse.ArgumentParser(
        "Plot the evolution of the density of a degree of freedom")
    mlxtk.plot.argparser.add_plotting_arguments(parser)
    parser.add_argument(
        "--in",
        type=str,
        dest="input_file",
        default="gpop",
        help="input_file (defaults to \"gpop\")")
    parser.add_argument(
        "--dof",
        type=int,
        default=1,
        help="degree of freedom for which to plot the density")
    args = parser.parse_args()

    global density
    global grid
    global times
    times, grids, densities = read_gpop(args.input_file)
    grid = grids[args.dof]
    density = densities[args.dof]

    line = [None]

    def init_plot(plot):
        global density
        global grid
        line[0] = plot.axes.plot(grid, density[0])[0]
        plot.axes.set_xlabel("$x$")
        plot.axes.set_ylabel("density")

    def update_plot(index):
        global density
        line[0].set_ydata(density[index])

    program = SimplePlotProgram(
        "Density of DOF {}".format(args.dof),
        init_plot,
        update_plot=update_plot)
    program.main(args)


if __name__ == "__main__":
    main()
