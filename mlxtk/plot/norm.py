from mlxtk.plot import plot
import numpy


def plot_norm(data, absolute=False):
    p = plot.Plot()
    if absolute:
        p.axes.plot(data.time, numpy.abs(1 - data.norm))
        p.axes.set_ylabel(r"$\left|1-\Psi(t)\right|$")
    else:
        p.axes.plot(data.time, 1 - data.norm)
        p.axes.set_ylabel(r"$1-\Psi(t)$")
    p.axes.set_xlabel("$t$")
    return p
