import h5py
import os
import sys

from mlxtk.inout import hdf5
from mlxtk.inout import expval

from PyQt5 import QtCore, QtGui, QtWidgets
from mlxtk.plot.qt5_plot import Qt5Plot
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


def list_expvals(path):
    parsed = hdf5.parse_hdf5_path(path)
    if not parsed:
        return list_expvals_ascii(path)
    else:
        return list_expvals_hdf5(parsed)


def list_expvals_ascii(path):
    expvals = []
    for entry in os.listdir(path):
        expval = os.path.join(path, entry)
        if not os.path.isfile(expval):
            continue
        if os.path.splitext(expval)[1] != ".expval":
            continue
        expvals.append(expval)
    return sorted(expvals)


def list_expvals_hdf5(parsed_path):
    path, path_inside = parsed_path
    expvals = []
    with h5py.File(path, "r") as fhandle:
        for group in fhandle[path_inside]:
            if group.startswith("expval_"):
                expvals.append(path + os.path.join(path_inside, group))
    return sorted(expvals)

def get_expval_name(path):
    tmp = os.path.basename(path)
    base, ext = os.path.splitext(path)
    if ext == ".expval":
        return base
    return tmp[7:]


def plot_expval(path, plot):
    data = expval.read_expval(path)
    plot.axes.plot(data.time, data.real)

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, path, **kwargs):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.main_widget = QtWidgets.QWidget()
        self.tab_widget = QtWidgets.QTabWidget(self.main_widget)

        self.expvals = list_expvals(path)
        self.tabs = [QtWidgets.QWidget(self.tab_widget) for expval in self.expvals]
        self.tab_layouts = []
        self.plots = []
        for expval, tab in zip(self.expvals, self.tabs):
            self.tab_layouts.append(QtWidgets.QVBoxLayout())
            self.tab_widget.addTab(tab, get_expval_name(expval))
            self.plots.append(Qt5Plot(tab))

            self.tab_layouts[-1].addWidget(self.plots[-1])
            tab.setLayout(self.tab_layouts[-1])

            plot_expval(expval, self.plots[-1])
            self.plots[-1].figure.tight_layout()
            self.plots[-1].updateGeometry()


        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.tab_widget)
        self.main_widget.setLayout(self.layout)
        self.setCentralWidget(self.main_widget)


def main():
    application = QtWidgets.QApplication(sys.argv)
    window = ApplicationWindow(sys.argv[1])
    window.show()
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
