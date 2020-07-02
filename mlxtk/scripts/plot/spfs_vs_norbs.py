import argparse
import sys

import numpy
from PySide2 import QtWidgets

from mlxtk import dvr, inout, plot
from mlxtk.scripts.dmat_evec_slider import DmatEvecSlider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_psi", nargs="?", help="path to psi file")
    parser.add_argument(
        "path_norbs", nargs="?", help="path to file with natural orbitals"
    )
    plot.add_argparse_2d_args(parser)
    args = parser.parse_args()

    times, spfs = inout.read_spfs(args.path_psi)
    _, grid, norbs = inout.read_dmat_evecs_grid(args.path_norbs)

    weights = numpy.sqrt(dvr.add_harmdvr(225, 0.0, 0.3).get_weights())

    for i in range(norbs.shape[0]):
        for j in range(norbs.shape[1]):
            norbs[i][j] *= weights

    app = QtWidgets.QApplication(sys.argv)
    window = DmatEvecSlider(times, grid, numpy.abs(spfs) - numpy.abs(norbs), args)
    assert window
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
