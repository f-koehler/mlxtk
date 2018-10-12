import argparse
import sys

from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QApplication, QGraphicsView, QLabel, QSlider

from ..inout.gpop import read_gpop_hdf5
from ..ui import load_ui
from ..ui.plot_widgets import SingleLinePlot


class GUI(QObject):
    def __init__(self, slider_values, x, ys, parent=None):
        super(GUI, self).__init__(parent)

        self.slider_values = slider_values
        self.x = x
        self.ys = ys

        # load UI
        self.window = load_ui("plot_slider.ui")
        print("Helo")

        # get UI elements
        self.plot = self.window.findChild(QGraphicsView, "plot")
        self.slider = self.window.findChild(QSlider, "slider_value")
        self.label_slider = self.window.findChild(QLabel, "label_slider")
        self.label_value = self.window.findChild(QLabel, "label_value")

        #     # init GUI elements
        #     self.slider.setMinimum(0)
        #     self.slider.setMaximum(len(slider_values) - 1)
        #     self.slider.setValue(0)
        #     self.slider.setSingleStep(1)
        #     self.slider.setPageStep(int(round(0.1 * len(slider_values))))
        #     self.slider.valueChanged.connect(self.update)
        #     self.label_value.setText(str(slider_values[0]))

        #     plot = SingleLinePlot(x, ys[0])
        #     replace_widget(self.plot, plot)
        #     self.plot = plot

        self.window.show()

    # def update(self, index):
    #     self.label_value = str(self.slider_values[index])
    #     self.plot.line.set_ydata(self.ys[index])
    #     self.plot.fig.draw_idle()


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", nargs="?", default="gpop.hdf5", help="path to the gpop file"
    )
    parser.add_argument("-d", "--dof", type=int, default=1, help="degree of freedom")

    args = parser.parse_args()

    # load data
    time, grid, density = read_gpop_hdf5(args.path, dof=args.dof)

    # create app
    app = QApplication(sys.argv)
    GUI(time, grid, density)

    # run app
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
