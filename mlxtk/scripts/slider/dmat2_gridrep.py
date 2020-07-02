import argparse
import sys

from PySide2 import QtWidgets

from mlxtk import plot
from mlxtk.inout.dmat2 import read_dmat2_gridrep
from mlxtk.scripts.slider import g1


class Gui(g1.Gui):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        type=str,
        default="dmat2.h5/dmat2_gridrep",
        help="path to the dmat file",
    )
    plot.add_argparse_2d_args(parser)
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    data = read_dmat2_gridrep(args.path)
    gui = Gui(data, 0, len(data[0]) - 1)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
