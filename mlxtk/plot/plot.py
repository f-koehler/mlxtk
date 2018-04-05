import matplotlib

if matplotlib.get_backend() == "nbAgg":
    from matplotlib import pyplot


    class Plot(object):
        def __init__(self, projection=None):
            self.figure, self.axes = pyplot.subplots(1, 1)

        def tight_layout(self):
            self.figure.tight_layout()

else:
    from matplotlib.backends import backend_agg as backend
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.figure


    class Plot(object):
        def __init__(self, projection=None):
            self.figure = matplotlib.figure.Figure(figsize=(10, 8))
            backend.FigureCanvas(self.figure)

            if projection is None:
                self.axes = self.figure.add_subplot(1, 1, 1)
            else:
                self.axes = self.figure.add_subplot(1, 1, 1, projection=projection)

        def tight_layout(self):
            self.figure.tight_layout()
