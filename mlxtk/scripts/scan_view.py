import argparse
import itertools
import os
import pickle
import subprocess
import sys

from PyQt5 import QtCore, QtWidgets
import h5py

from mlxtk.inout import hdf5

class TableVariables(QtWidgets.QTableView):
    def __init__(self, parent=None):
        QtWidgets.QTableView.__init__(self, parent)

        self.open_plot_function = lambda: None

    def keyPressEvent(self, event):
        key = event.key()
        if (key == QtCore.Qt.Key_Enter) or (key == QtCore.Qt.Key_Return):
            self.open_plot_function()
        else:
            QtWidgets.QTableView.keyPressEvent(self, event)


class DataModelVariables(QtCore.QAbstractTableModel):
    def __init__(self, scan_parameters, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)

        self.scan_parameters = scan_parameters

        self.table_values = list(
            itertools.product(*self.scan_parameters["values"]))

        self.variable_indices = [
            i for i, _ in enumerate(self.scan_parameters["values"])
            if len(self.scan_parameters["values"][i]) > 1
        ]

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.table_values)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.variable_indices) + 1

    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                if section == 0:
                    return "Index"
                return self.scan_parameters["names"][self.variable_indices[
                    section - 1]]

        return QtCore.QVariant()

    def data(self, index, role):
        row = index.row()
        col = index.column()

        if role == QtCore.Qt.DisplayRole:
            if col == 0:
                return row

            return self.table_values[row][self.variable_indices[col - 1]]

        return QtCore.QVariant()


class DataModelConstants(QtCore.QAbstractTableModel):
    def __init__(self, scan_parameters, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)

        self.scan_parameters = scan_parameters

        self.constant_indices = [
            i for i, _ in enumerate(self.scan_parameters["values"])
            if len(self.scan_parameters["values"][i]) == 1
        ]

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.constant_indices)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 1

    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                if section == 0:
                    return "Value"

            else:
                return self.scan_parameters["names"][self.constant_indices[
                    section]]

        return QtCore.QVariant()

    def data(self, index, role):
        row = index.row()

        if role == QtCore.Qt.DisplayRole:
            return self.scan_parameters["values"][self.constant_indices[row]][
                0]

        return QtCore.QVariant()


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, path):
        self.path = path

        if hdf5.is_hdf5_file(path):
            with h5py.File(path, "r") as fhandle:
                self.scan_parameters = pickle.loads(
                    bytes(fhandle.attrs["scan_parameters"]))

                self.subdirectories = [sim for sim in fhandle["sim_0"]]

            self.hdf5_mode = True
        else:
            with open(os.path.join(self.path, "parameters.pickle"),
                      "rb") as fhandle:
                self.scan_parameters = pickle.load(fhandle)

            self.hdf5_mode = False

        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("ScanView")

        self.tool_bar = self.addToolBar("main")

        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)
        self.tabs = QtWidgets.QTabWidget()
        self.tab_variables = QtWidgets.QWidget()
        self.tab_constants = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_variables, "Variables")
        self.tabs.addTab(self.tab_constants, "Constants")

        self.layout_main = QtWidgets.QVBoxLayout()
        self.layout_main.addWidget(self.tool_bar)
        self.layout_main.addWidget(self.tabs)
        self.main_widget.setLayout(self.layout_main)

        self.label_plot_type = QtWidgets.QLabel("Plot Type: ")
        self.tool_bar.addWidget(self.label_plot_type)
        self.combo_plot_type = QtWidgets.QComboBox()
        self.combo_plot_type.addItem("energy", "energy")
        self.combo_plot_type.addItem("gpop", "gpop")
        self.combo_plot_type.addItem("gpop_diff", "gpop_diff")
        self.combo_plot_type.addItem("gpop_slider", "gpop_slider")
        self.combo_plot_type.addItem("natpop", "natpop")
        self.combo_plot_type.addItem("norm", "norm")
        self.combo_plot_type.addItem("overlap", "overlap")
        self.tool_bar.addWidget(self.combo_plot_type)

        self.label_subdir = QtWidgets.QLabel("    Subdirectory: ")
        self.tool_bar.addWidget(self.label_subdir)
        self.combo_subdir = QtWidgets.QComboBox()
        self.combo_subdir.setEnabled(False)
        self.tool_bar.addWidget(self.combo_subdir)

        self.button_plot = QtWidgets.QPushButton("Plot")
        self.button_plot.pressed.connect(self.open_plot)
        self.tool_bar.addWidget(QtWidgets.QLabel("    "))
        self.tool_bar.addWidget(self.button_plot)

        self.model_variables = DataModelVariables(self.scan_parameters)
        self.model_variables_proxy = QtCore.QSortFilterProxyModel()
        self.model_variables_proxy.setSourceModel(self.model_variables)

        self.table_variables = TableVariables(self.tab_variables)
        self.table_variables.setSortingEnabled(True)
        self.table_variables.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.table_variables.setFocus()
        self.table_variables.setModel(self.model_variables_proxy)
        self.table_variables.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self.table_variables.doubleClicked.connect(self.open_plot)
        self.table_variables.clicked.connect(self.update_subdirs)
        self.table_variables.selectRow(0)
        self.table_variables.open_plot_function = self.open_plot

        self.model_constants = DataModelConstants(self.scan_parameters)
        self.table_constants = QtWidgets.QTableView(self.tab_constants)
        self.table_constants.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.table_constants.setModel(self.model_constants)

        self.layout_variables = QtWidgets.QVBoxLayout()
        self.layout_variables.addWidget(self.table_variables)
        self.tab_variables.setLayout(self.layout_variables)

        self.layout_constants = QtWidgets.QVBoxLayout()
        self.layout_constants.addWidget(self.table_constants)
        self.tab_constants.setLayout(self.layout_constants)

        self.update_subdirs(
            self.table_variables.selectionModel().selectedRows()[0])

    def update_subdirs(self, index):
        if self.combo_subdir.isEnabled():
            old_selection = self.combo_subdir.itemData(
                self.combo_subdir.currentIndex())
        else:
            old_selection = None
            self.combo_subdir.setEnabled(True)

        if self.hdf5_mode:
            with h5py.File(self.path, "r") as fhandle:
                subdirectories = [sim for sim in fhandle["sim_0"]]
        else:
            sim_dir = os.path.join(self.path, "sim_" + str(index.row()))
            subdirectories = [
                subdir for subdir in os.listdir(sim_dir)
                if os.path.isdir(os.path.join(sim_dir, subdir))
                and subdir not in ["states"]
            ]

        while self.combo_subdir.count() > 0:
            self.combo_subdir.removeItem(0)

        for subdir in subdirectories:
            self.combo_subdir.addItem(subdir, subdir)

        if old_selection in subdirectories:
            self.combo_subdir.setCurrentIndex(
                subdirectories.index(old_selection))

    def open_plot(self, index=None):
        if index is not None:
            indices = [index]
        else:
            indices = self.table_variables.selectionModel().selectedRows()

        which = self.combo_plot_type.itemData(
            self.combo_plot_type.currentIndex())

        indices = [
            int(self.table_variables.model().data(
                self.table_variables.model().index(index.row(), 0)))
            for index in indices
        ]

        if which == "gpop_diff":
            if len(indices) != 2:
                msg = QtWidgets.QErrorMessage()
                msg.showMessage("Please select exactly two simulations")
            else:
                self.plot_gpop_diff(indices[0], indices[1])
            return

        for index in indices:
            if which == "energy":
                self.plot_energy(index)
            elif which == "gpop":
                self.plot_gpop(index)
            elif which == "gpop_slider":
                self.plot_gpop_slider(index)
            elif which == "natpop":
                self.plot_natpop(index)
            elif which == "norm":
                self.plot_norm(index)
            elif which == "overlap":
                self.plot_overlap(index)

    def plot_energy(self, index):
        subdir = self.combo_subdir.itemData(self.combo_subdir.currentIndex())
        output_file = os.path.join(self.path, "sim_" + str(index), subdir,
                                   "output")
        print(output_file)
        subprocess.Popen(["plot_energy", "--in", output_file])

    def plot_gpop(self, index):
        subdir = self.combo_subdir.itemData(self.combo_subdir.currentIndex())
        output_file = os.path.join(self.path, "sim_" + str(index), subdir,
                                   "gpop")
        subprocess.Popen(["plot_gpop", "--in", output_file])

    def plot_gpop_diff(self, index1, index2):
        subdir = self.combo_subdir.itemData(self.combo_subdir.currentIndex())
        output_file1 = os.path.join(self.path, "sim_" + str(index1), subdir,
                                    "gpop")
        output_file2 = os.path.join(self.path, "sim_" + str(index2), subdir,
                                    "gpop")
        subprocess.Popen(
            ["plot_gpop_diff", "--in1", output_file1, "--in2", output_file2])

    def plot_gpop_slider(self, index):
        subdir = self.combo_subdir.itemData(self.combo_subdir.currentIndex())
        output_file = os.path.join(self.path, "sim_" + str(index), subdir,
                                   "gpop")
        subprocess.Popen(["plot_gpop_slider", "--in", output_file])

    def plot_natpop(self, index):
        subdir = self.combo_subdir.itemData(self.combo_subdir.currentIndex())
        output_file = os.path.join(self.path, "sim_" + str(index), subdir,
                                   "natpop")
        subprocess.Popen(["plot_natpop", "--in", output_file])

    def plot_norm(self, index):
        subdir = self.combo_subdir.itemData(self.combo_subdir.currentIndex())
        output_file = os.path.join(self.path, "sim_" + str(index), subdir,
                                   "output")
        subprocess.Popen(["plot_norm", "--in", output_file])

    def plot_overlap(self, index):
        subdir = self.combo_subdir.itemData(self.combo_subdir.currentIndex())
        output_file = os.path.join(self.path, "sim_" + str(index), subdir,
                                   "output")
        subprocess.Popen(["plot_overlap", "--in", output_file])


def main():
    parser = argparse.ArgumentParser(
        description="Explore the results of a parameter scan interactively")
    parser.add_argument(
        "path",
        type=str,
        help="path to a scan directory or a HDF5 file",
        default=".")
    args = parser.parse_args()

    application = QtWidgets.QApplication(sys.argv)
    window = ApplicationWindow(args.path)
    window.show()
    sys.exit(application.exec_())
