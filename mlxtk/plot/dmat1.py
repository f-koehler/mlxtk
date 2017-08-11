import mlxtk.plot.plot as plot

from matplotlib import pyplot
import numpy


def plot_dmat1(data, dvr=None):
    p = plot.Plot()
    p.activate()
    pyplot.title("dmat (time = {})".format(data.time[0]))
    n = len(data.x.unique().tolist())

    if dvr:
        p.axes.plot(
            data.y.values[0:n],
            (data.real.values.reshape(n, n).diagonal() +
             data.imaginary.values.reshape(n, n).diagonal()) / dvr.weights)
    else:
        p.axes.plot(data.y.values[0:n],
                    data.real.values.reshape(n, n).diagonal() +
                    data.imaginary.values.reshape(n, n).diagonal())

    p.axes.set_xlabel("$x$")
    p.axes.set_ylabel(r"${\left|\Psi(x)\right|}^2$")

    return p


def plot_dmat1_heatmap(data, dvr=None):
    p = plot.Plot()
    p.activate()
    pyplot.title("dmat (time = {})".format(data.time[0]))
    n = len(data.x.unique().tolist())

    if dvr:
        weights_x, weights_y = numpy.meshgrid(dvr.x, dvr.x)
        p.axes.pcolormesh(
            data.x.values.reshape(n, n),
            data.y.values.reshape(n, n),
            data.real.values.reshape(n, n) / (weights_x * weights_y))
    else:
        p.axes.pcolormesh(
            data.x.values.reshape(n, n),
            data.y.values.reshape(n, n), data.real.values.reshape(n, n))

    p.axes.set_xlabel("x")
    p.axes.set_ylabel("y")

    return p
