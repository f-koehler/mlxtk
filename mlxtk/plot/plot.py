import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot


class Plot:
    def __init__(self, **kwargs):
        figure = kwargs.get("figure", None)
        axes = kwargs.get("axes", None)

        if not figure or not axes:
            self.figure, self.axes = matplotlib.pyplot.subplots(
                nrows=kwargs.get("nrows", 1), ncols=kwargs.get("ncols", 1))
        else:
            self.figure = figure
            self.axes = None

    def activate(self):
        try:
            matplotlib.pyplot.sca(self.axes)
        except:
            try:
                matplotlib.pyplot.sca(self.axes[0])
            except:
                matplotlib.pyplot.sca(self.axes[0][0])

    def save(self, *args, **kwargs):
        self.figure.savefig(*args, **kwargs)

    def show(self):
        self.activate()
        matplotlib.pyplot.show()
