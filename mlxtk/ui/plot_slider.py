import argparse
from abc import ABC, abstractmethod
from typing import Optional

from mpl_toolkits.mplot3d import Axes3D
from PySide2 import QtWidgets

from mlxtk.ui import MatplotlibWidget, load_ui


class PlotSlider(QtWidgets.QWidget):
    def __init__(
        self,
        min_index: int,
        max_index: int,
        projection: str = None,
        plot_args: Optional[argparse.Namespace] = None,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)

        self.projection = projection
        self.plot_args = plot_args

        # fetch widgets
        self.window: QtWidgets.QMainWindow = load_ui("plot_slider.ui")
        self.plot: MatplotlibWidget = self.window.findChild(MatplotlibWidget, "plot")
        self.slider: QtWidgets.QSlider = self.window.findChild(
            QtWidgets.QSlider, "slider"
        )
        self.spin: QtWidgets.QSpinBox = self.window.findChild(
            QtWidgets.QSpinBox, "spin"
        )
        self.label: QtWidgets.QLabel = self.window.findChild(QtWidgets.QLabel, "label")

        # set up axes
        self.axes = self.plot.figure.add_subplot(1, 1, 1, projection=self.projection)
        self.plot.figure.set_tight_layout(True)

        # set up the slider
        self.slider.setTracking(True)
        self.slider.setMinimum(min_index)
        self.slider.setMaximum(max_index)
        self.slider.setValue(min_index)
        self.slider.valueChanged.connect(self.spin.setValue)

        # set up spin box
        self.spin.setMinimum(min_index)
        self.spin.setMaximum(max_index)
        self.spin.setValue(min_index)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.spin.valueChanged.connect(self.update_plot)

        self.init_plot()
        self.window.show()

    @abstractmethod
    def init_plot(self):
        pass

    @abstractmethod
    def update_plot(self, index: int):
        pass
