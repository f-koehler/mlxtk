import argparse
import sys

import numpy
from matplotlib.collections import QuadMesh
from PySide2 import QtWidgets

from mlxtk import inout, plot
from mlxtk.ui import MatplotlibWidget, load_ui


class DmatSlider(QtWidgets.QWidget):
    def __init__(self, times, dmat, plot_args, parent=None):
        super().__init__(parent)

        self.times = times
        self.index1 = numpy.arange(0, dmat.shape[1])
        self.index2 = numpy.arange(0, dmat.shape[2])
        self.dmat = dmat
        self.plot_args = plot_args
        self.time_index = 0
        self.mesh = None  # type: QuadMesh

        self.window = load_ui("dmat_spf_slider.ui")

        self.plot = self.window.findChild(
            MatplotlibWidget, "plot"
        )  # type: MatplotlibWidget
        self.slider_time = self.window.findChild(
            QtWidgets.QSlider, "slider_time"
        )  # type: QtWidgets.QSlider
        self.spin_time = self.window.findChild(
            QtWidgets.QSpinBox, "spin_time"
        )  # type: QtWidgets.QSpinBox
        self.label_time = self.window.findChild(
            QtWidgets.QLabel, "label_time"
        )  # type: QtWidgets.QLabel

        self.axes = self.plot.figure.subplots(1, 1)
        self.plot.figure.set_tight_layout(True)

        self.slider_time.setTracking(True)
        self.slider_time.setMaximum(len(self.times) - 1)
        self.slider_time.valueChanged.connect(self.spin_time.setValue)

        self.spin_time.setMaximum(len(self.times) - 1)
        self.spin_time.valueChanged.connect(self.slider_time.setValue)
        self.spin_time.valueChanged.connect(self.set_time)
        self.spin_time.valueChanged.connect(self.update_plot)

        self.set_time(self.time_index)
        self.init_plot()

        self.window.show()

    def set_time(self, index: int):
        self.time_index = index
        self.label_time.setText("Time: {:.4E}".format(self.times[index]))

    def init_plot(self):
        if self.mesh:
            self.mesh.remove()
            self.mesh = None

        X2, X1 = numpy.meshgrid(self.index2, self.index1)
        self.mesh = self.axes.pcolormesh(
            X1,
            X2,
            numpy.abs(self.dmat[self.time_index]),
            cmap="gnuplot",
            rasterized=True,
        )
        plot.apply_2d_args(self.axes, self.plot.figure, self.plot_args)
        self.plot.canvas.draw()

    def update_plot(self, index: int):
        self.mesh.set_array(self.dmat2[index, :-1, :-1].ravel())
        self.plot.canvas.draw()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to the dmat file")
    plot.add_argparse_2d_args(parser)
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    DmatSlider(*inout.read_dmat_spfrep(args.path), args)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
