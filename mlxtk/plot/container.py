import matplotlib.figure
import matplotlib.axes
import matplotlib.pyplot

class PlotContainer:
    def __init__(self, figure, axes):
        self.figure = figure
        self.axes = axes

    def activate(self):
        if isinstance(self.axes, matplotlib.axes.Axes):
            matplotlib.pyplot.sca(self.axes)
        else:
            if isinstance(self.axes[0], matplotlib.axes.Axes):
                matplotlib.pyplot.sca(self.axes[0])
            else:
                matplotlib.pyplot.sca(self.axes[0][0])

def show(con):
    con.activate()
    matplotlib.pyplot.show()
