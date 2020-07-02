import argparse
import sys
from typing import Optional, Tuple

import matplotlib
import numpy
from PySide2 import QtWidgets

from mlxtk import plot
from mlxtk.inout.dmat import read_dmat_spfrep_hdf5
from mlxtk.ui.plot_slider import PlotSlider


class Gui(PlotSlider):
    def __init__(
        self,
        data: Tuple[numpy.ndarray, numpy.ndarray],
        min_index: int,
        max_index: int,
        projection: str = None,
        plot_args: Optional[argparse.Namespace] = None,
        parent: QtWidgets.QWidget = None,
    ):
        self.time, self.values = data
        self.indices1 = list(range(self.values.shape[1]))
        self.indices2 = list(range(self.values.shape[2]))
        self.values = numpy.abs(self.values)
        self.mesh: matplotlib.collections.QuadMesh = None
        super().__init__(min_index, max_index, plot_args=plot_args, parent=parent)

        self.spin.valueChanged.connect(self.update_label)

    def init_plot(self):
        self.label.setText("Time: {:6.2f}".format(self.time[0]))

        if self.mesh:
            self.mesh.remove()
            self.mesh = None

        if self.plot_args:
            plot.apply_2d_args(self.axes, self.plot.figure, self.plot_args)

        self.axes.set_xlabel("i")
        self.axes.set_ylabel("j")
        self.axes.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        self.axes.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        X2, X1 = numpy.meshgrid(self.indices2, self.indices1)
        self.mesh = self.axes.pcolormesh(
            X1 - 0.5, X2 - 0.5, self.values[0], cmap="gnuplot", rasterized=True
        )
        self.plot.canvas.draw()

    def update_plot(self, index: int):
        self.mesh.set_array(self.values[index, :-1, :-1].ravel())
        self.mesh.set_clim(vmin=self.values[index].min(), vmax=self.values[index].max())
        self.plot.canvas.draw()

    def update_label(self, index: int):
        self.label.setText("Time: {:6.2f}".format(self.time[index]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to the dmat file")
    plot.add_argparse_2d_args(parser)
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    data = read_dmat_spfrep_hdf5(args.path, "dmat_spfrep")
    gui = Gui(data, 0, len(data[0]) - 1)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
