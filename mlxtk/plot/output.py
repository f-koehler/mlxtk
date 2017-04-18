import matplotlib.pyplot

import mlxtk.inout.output
import mlxtk.plot.container

def plot_overview(path):
    data = mlxtk.inout.output.read(path)

    fig, axes = matplotlib.pyplot.subplots(3)
    container = mlxtk.plot.container.PlotContainer(fig, axes)

    matplotlib.pyplot.sca(axes[0])
    matplotlib.pyplot.xlabel("$t$")
    matplotlib.pyplot.ylabel(r"$\left|\Psi\right|(t)-1$")
    matplotlib.pyplot.plot(data["time"], data["norm"]-1)

    matplotlib.pyplot.sca(axes[1])
    matplotlib.pyplot.xlabel("$t$")
    matplotlib.pyplot.ylabel(r"$\langle H\rangle (t)$")
    matplotlib.pyplot.plot(data["time"], data["energy"])

    matplotlib.pyplot.sca(axes[2])
    matplotlib.pyplot.xlabel("$t$")
    matplotlib.pyplot.ylabel(r"overlap")
    matplotlib.pyplot.plot(data["time"], data["overlap"])

    return container
