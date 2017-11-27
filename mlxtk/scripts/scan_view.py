import itertools
import os
import pickle
import subprocess
import sys

from PyQt5 import QtCore, QtWidgets


class DataModel(QtCore.QAbstractTableModel):
    def __init__(self, scan_directory, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)

        with open(os.path.join(scan_directory, "parameters.pickle"),
                  "rb") as fhandle:
            self.parameter_pickle = pickle.load(fhandle)

        self.table_values = list(
            itertools.product(*self.parameter_pickle["values"]))

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.table_values)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.parameter_pickle["names"]) + 1

    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                if section == 0:
                    return "Index"
                return self.parameter_pickle["names"][section - 1]

        return QtCore.QVariant()

    def data(self, index, role):
        row = index.row()
        col = index.column()

        if role == QtCore.Qt.DisplayRole:
            if col == 0:
                return row

            return self.table_values[row][col - 1]

        return QtCore.QVariant()


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, scan_directory):
        self.scan_directory = scan_directory
        sim_dir = os.path.join(self.scan_directory, "sim_0")
        self.subdirectories = [
            subdir for subdir in os.listdir(sim_dir)
            if os.path.isdir(os.path.join(sim_dir, subdir))
            and subdir not in ["states"]
        ]

        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("ScanView")

        self.main_widget = QtWidgets.QWidget()

        self.tool_bar = self.addToolBar("main")

        self.tool_bar.addWidget(QtWidgets.QLabel("Plot Type:"))

        self.combo_plot_type = QtWidgets.QComboBox()
        self.combo_plot_type.addItem("energy", "energy")
        self.combo_plot_type.addItem("gpop", "gpop")
        self.combo_plot_type.addItem("natpop", "natpop")
        self.combo_plot_type.addItem("norm", "norm")
        self.combo_plot_type.addItem("overlap", "overlap")
        self.tool_bar.addWidget(self.combo_plot_type)

        self.tool_bar.addWidget(QtWidgets.QLabel("Subdirectory:"))
        self.combo_subdir = QtWidgets.QComboBox()
        for subdir in self.subdirectories:
            self.combo_subdir.addItem(subdir, subdir)
        self.tool_bar.addWidget(self.combo_subdir)

        self.model = DataModel(scan_directory)

        self.table = QtWidgets.QTableView(self.main_widget)
        self.table.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.table.setFocus()
        self.setCentralWidget(self.table)

        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        self.table.doubleClicked.connect(self.open_plot)

    def open_plot(self, index):
        which = self.combo_plot_type.itemData(
            self.combo_plot_type.currentIndex())

        if which == "energy":
            self.plot_energy(index)
        elif which == "gpop":
            self.plot_gpop(index)
        elif which == "natpop":
            self.plot_natpop(index)
        elif which == "norm":
            self.plot_norm(index)
        elif which == "overlap":
            self.plot_overlap(index)

    def plot_energy(self, index):
        subdir = self.combo_subdir.itemData(self.combo_subdir.currentIndex())
        output_file = os.path.join(self.scan_directory,
                                   "sim_" + str(index.row()), subdir, "output")
        subprocess.Popen(["plot_energy", "--in", output_file])

    def plot_gpop(self, index):
        subdir = self.combo_subdir.itemData(self.combo_subdir.currentIndex())
        output_file = os.path.join(self.scan_directory,
                                   "sim_" + str(index.row()), subdir, "gpop")
        subprocess.Popen(["plot_gpop", "--in", output_file])

    def plot_natpop(self, index):
        subdir = self.combo_subdir.itemData(self.combo_subdir.currentIndex())
        output_file = os.path.join(self.scan_directory,
                                   "sim_" + str(index.row()), subdir, "natpop")
        subprocess.Popen(["plot_natpop", "--in", output_file])

    def plot_norm(self, index):
        subdir = self.combo_subdir.itemData(self.combo_subdir.currentIndex())
        output_file = os.path.join(self.scan_directory,
                                   "sim_" + str(index.row()), subdir, "output")
        subprocess.Popen(["plot_norm", "--in", output_file])

    def plot_overlap(self, index):
        subdir = self.combo_subdir.itemData(self.combo_subdir.currentIndex())
        output_file = os.path.join(self.scan_directory,
                                   "sim_" + str(index.row()), subdir, "output")
        subprocess.Popen(["plot_overlap", "--in", output_file])


def main():
    application = QtWidgets.QApplication(sys.argv)
    window = ApplicationWindow(".")
    window.show()
    sys.exit(application.exec_())
