from matplotlib import rc
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure
from matplotlib.pyplot import figure, show


def set_style():
    rc("axes", grid=True)
    rc("figure", dpi=200)
    rc("savefig", bbox="tight", directory="", dpi=600, format="pdf")


set_style()


class Plot:
    def __init__(self, **kwargs):
        self.figure = kwargs.get("figure", figure())
        # self.canvas = FigureCanvas(self.figure)
        self.axes = kwargs.get("axes", None)

        if not self.axes:
            self.axes = self.figure.add_subplot(1, 1, 1)

    def save(self, *args, **kwargs):
        self.figure.savefig(*args, **kwargs)

    def show(self):
        # self.figure.show()
        show()
