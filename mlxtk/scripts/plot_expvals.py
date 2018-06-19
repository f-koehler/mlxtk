import h5py
import os
import sys

from mlxtk.inout import hdf5
from mlxtk.inout import expval

from mlxtk import mpl
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


def plot_expval(path, plot, real=True, imaginary=False):
    data = expval.read_expval(path)

    lines = []

    if real:
        lines += plot.axes.plot(
            data.time,
            data.real,
            color="C0",
            label=r"$\mathrm{Re}\left[\left<O\right>\right](t)$",
        )
        plot.axes.set_ylabel(r"$\mathrm{Re}\left[\left<O\right>\right](t)$")

    if imaginary:
        ax = plot.axes.twinx() if real else plot.axes
        lines += ax.plot(
            data.time,
            data.imaginary,
            color="C1",
            label=r"$\mathrm{Im}\left[\left<O\right>\right](t)$",
        )
        ax.set_ylabel(r"$\mathrm{Im}\left[\left<O\right>\right](t)$")

    plot.axes.set_xlabel("$t$")
    if lines:
        plot.axes.legend(lines, [l.get_label() for l in lines])
    plot.figure.tight_layout()
    plot.updateGeometry()


class ExpvalTab(QtWidgets.QWidget):
    def __init__(self, path, parent):
        QtWidgets.QWidget.__init__(self)

        self.path = path

        self.plot = Qt5Plot(self)
        plot_expval(path, self.plot)

        self.toolbar = NavigationToolbar(self.plot, self)

        self.widget_controls = QtWidgets.QWidget(self)

        self.check_real = QtWidgets.QCheckBox("real part",
                                              self.widget_controls)
        self.check_real.setCheckState(2)
        self.check_imaginary = QtWidgets.QCheckBox("imaginary part",
                                                   self.widget_controls)

        self.layout_controls = QtWidgets.QGridLayout(self.widget_controls)
        self.layout_controls.addWidget(self.check_real, 0, 0)
        self.layout_controls.addWidget(self.check_imaginary, 0, 1)
        self.widget_controls.setLayout(self.layout_controls)

        def replot(dummy):
            self.layout.removeWidget(self.toolbar)
            self.layout.removeWidget(self.widget_controls)
            self.layout.removeWidget(self.plot)
            self.plot = Qt5Plot(self)
            self.toolbar = NavigationToolbar(self.plot, self)
            self.layout.addWidget(self.toolbar)
            self.layout.addWidget(self.widget_controls)
            self.layout.addWidget(self.plot)
            plot_expval(
                self.path,
                self.plot,
                self.check_real.isChecked(),
                self.check_imaginary.isChecked(),
            )

        self.check_real.stateChanged.connect(replot)
        self.check_imaginary.stateChanged.connect(replot)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.widget_controls)
        self.layout.addWidget(self.plot)
        self.setLayout(self.layout)

        if hasattr(parent, "addTab"):
            parent.addTab(self, get_expval_name(path))


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, path, **kwargs):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.main_widget = QtWidgets.QWidget()

        self.expvals = list_expvals(path)
        if len(self.expvals) == 1:
            self.tab_widget = None
            self.tabs = [ExpvalTab(self.expvals[0], self.main_widget)]
        else:
            self.tab_widget = QtWidgets.QTabWidget(self.main_widget)
            self.tabs = [
                ExpvalTab(expval, self.tab_widget) for expval in self.expvals
            ]

        self.layout = QtWidgets.QVBoxLayout()
        if len(self.expvals) == 1:
            self.layout.addWidget(self.tabs[0])
        else:
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
