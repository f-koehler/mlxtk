import argparse
import sys

import h5py
import numpy
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import (QApplication, QGraphicsView, QHeaderView, QLabel,
                             QListWidget, QTableWidget, QTabWidget)

from ..tools.diagonalize import diagonalize_1b_operator
from ..ui import load_ui, replace_widget
from ..ui.plot_widgets import SingleLinePlot


class SpectrumPlot(SingleLinePlot):
    def __init__(self, x, y, parent=None, **kwargs):
        kwargs["linestyle"] = " "
        kwargs["xlabel"] = "Index of Eigenstate"
        kwargs["ylabel"] = "$E_i$"
        super(SpectrumPlot, self).__init__(x, y, parent, **kwargs)

        self.vline = None
        self.hline = None

    def mark(self, x, y):
        if self.vline:
            self.vline.remove()
            del self.vline

        if self.hline:
            self.hline.remove()
            del self.hline

        self.vline = self.axes.axvline(x, color="C7")
        self.hline = self.axes.axhline(y, color="C7")
        self.draw_idle()


class GUI(QObject):
    def __init__(self,
                 min_index,
                 max_index,
                 grid,
                 weights,
                 energies,
                 spfs,
                 parent=None):
        super(GUI, self).__init__(parent)

        self.min_index = min_index
        self.max_index = max_index
        self.indices = list(range(min_index, max_index + 1))
        self.grid = grid
        self.weights = weights
        self.energies = energies
        self.spfs = spfs

        # load UI
        self.window = load_ui("spectrum_1b.ui")

        # get UI elements
        self.spf_list = self.window.findChild(QListWidget, "list_spfs")
        self.label_energy = self.window.findChild(QLabel, "label_energy")
        self.plot_spectrum = self.window.findChild(QGraphicsView,
                                                   "plot_spectrum")
        self.plot_abs = self.window.findChild(QGraphicsView, "plot_abs")
        self.plot_real = self.window.findChild(QGraphicsView, "plot_real")
        self.plot_imag = self.window.findChild(QGraphicsView, "plot_imag")
        self.table_properties = self.window.findChild(QTableWidget,
                                                      "table_properties")
        self.tabs = self.window.findChild(QTabWidget, "tabs")
        self.tab_spectrum = self.window.findChild(QTabWidget, "tab_spectrum")

        # caches for plots and values
        self.cache_plot_abs = {}
        self.cache_plot_real = {}
        self.cache_plot_imag = {}
        self.cache_max_overlap = {}

        # add items to list
        for i, E in enumerate(self.energies):
            self.spf_list.addItem("{}: E={:.4f}".format(i + min_index, E))

        # init GUI elements with first eigenstate
        self.spf_list.setCurrentItem(self.spf_list.item(0))
        self.spf_list.currentRowChanged.connect(self.select_item)
        self.update_table(0)
        self.table_properties.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.tabs.currentChanged.connect(self.switch_tab)

        plot = SpectrumPlot(self.indices, self.energies)
        replace_widget(self.plot_spectrum, plot)
        self.plot_spectrum = plot
        plot.mark(min_index, self.energies[0])

        self.set_abs_plot(0)
        self.set_real_plot(0)
        self.set_imag_plot(0)

        self.window.show()

    def set_abs_plot(self, index):
        if index not in self.cache_plot_abs:
            self.cache_plot_abs[index] = SingleLinePlot(
                self.grid,
                numpy.abs(self.spfs[index]),
                xlabel="$x$",
                ylabel=r"${\left|\varphi_{" + str(index) + r"}(x)\right|}^2$",
            )
        replace_widget(self.plot_abs, self.cache_plot_abs[index])
        self.plot_abs = self.cache_plot_abs[index]

    def set_real_plot(self, index):
        if index not in self.cache_plot_real:
            self.cache_plot_real[index] = SingleLinePlot(
                self.grid,
                numpy.real(self.spfs[index]),
                xlabel="$x$",
                ylabel=r"$\mathrm{Re}\left[\varphi_{" + str(index) +
                r"}(x)\right]$",
            )
        replace_widget(self.plot_real, self.cache_plot_real[index])
        self.plot_real = self.cache_plot_real[index]

    def set_imag_plot(self, index):
        if index not in self.cache_plot_imag:
            self.cache_plot_imag[index] = SingleLinePlot(
                self.grid,
                numpy.imag(self.spfs[index]),
                xlabel="$x$",
                ylabel=r"$\mathrm{Im}\left[\varphi_{" + str(index) +
                r"}(x)\right]$",
            )
        replace_widget(self.plot_imag, self.cache_plot_imag[index])
        self.plot_imag = self.cache_plot_imag[index]

    def update_table(self, index):
        self.table_properties.item(0, 0).setText(str(self.energies[index]))
        self.table_properties.item(1, 0).setText(str(index + self.min_index))
        self.table_properties.item(2, 0).setText(
            str(
                numpy.vdot(self.spfs[index] * numpy.sqrt(self.weights),
                           self.spfs[index])))

    def update(self, index, tab_index):
        if tab_index == 0:
            self.plot_spectrum.mark(index + self.min_index,
                                    self.energies[index])
        elif tab_index == 1:
            self.set_abs_plot(index)
        elif tab_index == 2:
            self.set_real_plot(index)
        elif tab_index == 3:
            self.set_imag_plot(index)
        else:
            self.update_table(index)

    def select_item(self, index):
        tab_index = self.tabs.currentIndex()
        self.update(index, tab_index)

    def switch_tab(self, tab_index):
        index = self.spf_list.currentRow()
        self.update(index, tab_index)


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("path",
                        help="path to the one body operator matrix file")
    parser.add_argument(
        "--min",
        dest="min_",
        type=int,
        default=0,
        help="minimum index of the eigenstate",
    )
    parser.add_argument(
        "--max",
        dest="max_",
        type=int,
        default=None,
        help="maximum index of the eigenstate",
    )

    args = parser.parse_args()

    # load operator matrix
    with h5py.File(args.path, "r") as fp:
        matrix = fp["matrix"][:, :]
        grid = fp["grid_1"][:]
        weights = fp["weights_1"][:]

    # diagonalize
    energies, spfs = diagonalize_1b_operator(matrix, matrix.shape[0])
    min_ = args.min_
    max_ = args.max_
    if max_ is None:
        max_ = 10

    # create app
    app = QApplication(sys.argv)
    GUI(min_, max_, grid, weights, energies[min_:max_ + 1],
        spfs[min_:max_ + 1])

    # run app
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
