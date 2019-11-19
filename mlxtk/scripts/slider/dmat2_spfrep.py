import argparse
import sys
from typing import Optional, Tuple

import matplotlib
import numpy
from PySide2 import QtWidgets

from mlxtk import plot
from mlxtk.inout.dmat2 import read_dmat2_spfrep_hdf5
from mlxtk.scripts.slider import dmat_spfrep
from mlxtk.ui.plot_slider import PlotSlider


class Gui(dmat_spfrep.Gui):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to the dmat file")
    plot.add_argparse_2d_args(parser)
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    data = read_dmat2_spfrep_hdf5(args.path, "dmat2_spfrep")
    gui = Gui(data, 0, len(data[0]) - 1)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
