import numpy
from matplotlib.axes import Axes


def plot_energy(ax: Axes, time: numpy.ndarray, energy: numpy.ndarray,
                **kwargs):
    ax.plot(time, energy, label=kwargs.get("label"))
    ax.set_xlabel("$t$")
    ax.set_ylabel("$E(t)$")


def plot_energy_diff(ax: Axes, time: numpy.ndarray, energy1: numpy.ndarray,
                     energy2: numpy.ndarray):
    ax.plot(time, energy1 - energy2)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$E(t)-E^\prime(t)$")
