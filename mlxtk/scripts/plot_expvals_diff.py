import numpy
import sys

from mlxtk.inout import expval

from mlxtk import mpl
from PyQt5 import QtCore, QtWidgets
from mlxtk.plot.qt5_plot import Qt5Plot
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from mlxtk.scripts.plot_expvals import list_expvals, get_expval_name


def plot_expval_diff(path1, path2, plot, real=True, imaginary=False):
    data1 = expval.read_expval(path1)
    data2 = expval.read_expval(path2)

    assert numpy.array_equal(data1.time, data2.time)

    lines = []

    if real:
        lines += plot.axes.plot(
            data1.time,
            data1.real - data2.real,
            color="C0",
            label=
            r"$\mathrm{Re}\left[{\left<O\right>}_1\right](t)-\mathrm{Re}\left[{\left<O\right>}_2\right](t)$",
        )
        plot.axes.set_ylabel(
            r"$\mathrm{Re}\left[{\left<O\right>}_1\right](t)-\mathrm{Re}\left[{\left<O\right>}_2\right](t)$"
        )

    if imaginary:
        ax = plot.axes.twinx() if real else plot.axes
        lines += ax.plot(
            data1.time,
            data1.imaginary - data2.imaginary,
            color="C1",
            label=
            r"$\mathrm{Im}\left[{\left<O\right>}_1\right](t)-\mathrm{Im}\left[{\left<O\right>}_2\right](t)$",
        )
        ax.set_ylabel(
            r"$\mathrm{Im}\left[{\left<O\right>}_1\right](t)-\mathrm{Im}\left[{\left<O\right>}_2\right](t)$"
        )

    plot.axes.set_xlabel("$t$")
    if lines:
        plot.axes.legend(lines, [l.get_label() for l in lines])
    plot.figure.tight_layout()
    plot.updateGeometry()


class ExpvalDiffTab(QtWidgets.QWidget):
    def __init__(self, path1, path2, parent):
        QtWidgets.QWidget.__init__(self)

        self.path1 = path1
        self.path2 = path2

        self.plot = Qt5Plot(self)
        plot_expval_diff(path1, path2, self.plot)

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
            plot_expval_diff(
                self.path1,
                self.path2,
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
            parent.addTab(self, get_expval_name(path1))


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, path1, path2, **kwargs):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.main_widget = QtWidgets.QWidget()

        tmp = [(exp1, exp2)
               for exp1, exp2 in zip(list_expvals(path1), list_expvals(path2))
               if get_expval_name(exp1) == get_expval_name(exp2)]
        self.expvals1 = [t[0] for t in tmp]
        self.expvals2 = [t[1] for t in tmp]

        if len(self.expvals1) == 1:
            self.tab_widget = None
            self.tabs = [
                ExpvalDiffTab(self.expvals1[0], self.expvals2[0],
                              self.main_widget)
            ]
        else:
            self.tab_widget = QtWidgets.QTabWidget(self.main_widget)
            self.tabs = [
                ExpvalDiffTab(exp1, exp2, self.tab_widget)
                for exp1, exp2 in zip(self.expvals1, self.expvals2)
            ]

        self.layout = QtWidgets.QVBoxLayout()
        if len(self.expvals1) == 1:
            self.layout.addWidget(self.tabs[0])
        else:
            self.layout.addWidget(self.tab_widget)
        self.main_widget.setLayout(self.layout)
        self.setCentralWidget(self.main_widget)


def main():
    application = QtWidgets.QApplication(sys.argv)
    window = ApplicationWindow(sys.argv[1], sys.argv[2])
    window.show()
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
