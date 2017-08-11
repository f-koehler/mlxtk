from mlxtk.plot import plot


def plot_energy(data, **kwargs):
    p = plot.Plot()
    if kwargs.get("logx", False):
        p.axes.set_xscale("log")
    if kwargs.get("logy", False):
        p.axes.set_yscale("log")
    p.axes.plot(data.time, data.energy)
    p.axes.set_xlabel("$t$")
    p.axes.set_ylabel(r"$\langle H\rangle (t)$")
    return p
