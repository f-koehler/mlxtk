import numpy
from matplotlib.axes import Axes


def plot_expval(ax: Axes, time: numpy.ndarray, expval: numpy.ndarray, **kwargs):
    if numpy.abs(numpy.imag(expval)).max() > 1e-10:
        raise ValueError("expectation value as considerable imaginary part")
    ax.plot(time, numpy.real(expval))
    ax.set_xlabel("$t$")

    if "symbol" in kwargs:
        ax.set_ylabel(r"$\left<" + symbol + "\right>(t)$")
    else:
        ax.set_ylabel(r"$\left<O\right>(t)$")
