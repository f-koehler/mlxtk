import matplotlib.animation
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot

class PlotContainer:
    def __init__(self, figure=None, axes=None):
        if not figure:
            self.figure, self.axes = matplotlib.pyplot.subplots(nrows=1, ncols=1)
        else:
            self.figure = figure
            self.axes = axes
        self.animation = None

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

def save_animation(con, path):
    con.animation.save(
        path,
        writer="ffmpeg",
        fps="30",
        dpi=150,
        codec="h264"
    )
