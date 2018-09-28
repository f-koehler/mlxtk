import numpy


def plot_energy(ax, time, energy):
    ax.plot(time, energy)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$E(t)$")
