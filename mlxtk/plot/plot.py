# import matplotlib
# import matplotlib.figure
# import matplotlib.pyplot
# import importlib

# def get_backend_module():
#     backend = matplotlib.get_backend().lower()
#     module = importlib.import_module("matplotlib.backends.backend_"
#         + backend)
#     return module

# def get_canvas_type():
#     return get_backend_module().FigureCanvas

# class Plot():
#     def __init__(self):
#         self.figure = matplotlib.pyplot.figure()
#         # self.canvas = get_canvas_type()(self.figure)

#     def save(self, *args, **kwargs):
#         self.figure.savefig(*args, **kwargs)

#     def show(self):
#         self.figure.show()

import matplotlib.figure
from matplotlib.backends import backend_agg as backend


class Plot(object):
    def __init__(self):
        self.figure = matplotlib.figure.Figure(figsize=(10, 8))
        backend.FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(1, 1, 1)

    def tight_layout(self):
        self.figure.tight_layout()
