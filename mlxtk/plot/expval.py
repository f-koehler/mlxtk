import numpy
from matplotlib.axes import Axes


def plot_expval(ax: Axes, time: numpy.ndarray, expval: numpy.ndarray,
                **kwargs):
    if numpy.abs(numpy.imag(expval)).max() > 1e-10:
        raise ValueError("expectation value as considerable imaginary part")
    ax.plot(time, numpy.real(expval), label=kwargs.get("label"))
    ax.set_xlabel("$t$")

    ax.set_ylabel(r"$\left<" + kwargs.get("symbol", "O") + r"\right>(t)$")
