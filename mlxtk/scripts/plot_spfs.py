import argparse
import sys

from PySide2 import QtWidgets

from .. import inout, plot
from .dmat_evec_slider import DmatEvecSlider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path",
                        nargs="?",
                        default="psi",
                        help="path to psi file")
    parser.add_argument(
        "gpop",
        nargs="?",
        default="gpop",
        help="path to a gpop file in order to determin the grid")
    plot.add_argparse_2d_args(parser)
    args = parser.parse_args()

    times, spfs = inout.read_spfs(args.path)
    _, grid, _ = inout.read_gpop(args.gpop, dof=1)

    app = QtWidgets.QApplication(sys.argv)
    window = DmatEvecSlider(times, grid, spfs, args)
    assert window
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
