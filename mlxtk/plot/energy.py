def plot_energy(plot, data):
    plot.axes.plot(data.time, data.energy)
    plot.axes.set_xlabel("$t$")
    plot.axes.set_ylabel(r"$\langle H\rangle (t)$")
