import numpy
from matplotlib.axes import Axes


def plot_entropy(ax: Axes, time: numpy.ndarray, entropy: numpy.ndarray,
                 **kwargs):
    ax.plot(time, entropy, label=kwargs.get("label"))
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$S_{\mathrm{B}}(t)$")


def plot_entropy_diff(ax: Axes, time: numpy.ndarray, entropy1: numpy.ndarray,
                      entropy2: numpy.ndarray):
    ax.plot(time, entropy1 - entropy2)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$S_{\mathrm{B}}(t)-S_{\mathrm{B}}^\prime(t)$")
