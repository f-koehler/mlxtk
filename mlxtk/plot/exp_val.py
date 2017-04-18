import matplotlib.pyplot

import mlxtk.inout.exp_val
import mlxtk.plot.container


def plot_real(file, container=None, symbol="O"):
    if not container:
        figure, axis = matplotlib.pyplot.subplots()
        container = plot.container.PlotContainer(figure, axis)
        matplotlib.pyplot.xlabel("$t$")
        matplotlib.pyplot.ylabel(r"$\mathrm{Re}\,\langle O \rangle$")

    data = inout.exp_val.read(file)
    matplotlib.pyplot.plot(data["time"], data["real"])

    return container
