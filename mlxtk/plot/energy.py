import numpy
from matplotlib.axes import Axes


def plot_energy(ax: Axes, time: numpy.ndarray, energy: numpy.ndarray):
    ax.plot(time, energy)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$E(t)$")
