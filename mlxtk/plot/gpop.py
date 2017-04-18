import matplotlib.pyplot
import os
import re
import numpy

import mlxtk.inout.gpop
import mlxtk.plot.container

def plot_overview(dir, ncols=1):
    dir = os.path.expanduser(dir)
    re_file = re.compile(r"^density_(\d+)\.gz$")

    files = []
    ids = []
    for entry in os.listdir(dir):
        path = os.path.join(dir, entry)
        if not os.path.isfile(path):
            continue

        m = re_file.match(entry)
        if not m:
            continue

        ids.append(int(m.group(1)))

    n = len(ids)
    ncols = min(n, ncols)
    nrows = round(float(n) / ncols)

    fig, axes = matplotlib.pyplot.subplots(nrows=nrows, ncols=ncols)
    container = mlxtk.plot.container.PlotContainer(fig, axes)

    for id in ids:
        grid, density = mlxtk.inout.gpop.read(dir, id)

        if nrows == 1:
            if ncols == 1:
                matplotlib.pyplot.sca(axes)
            else:
                matplotlib.pyplot.sca(axes[i])
        elif ncols == 1:
            matplotlib.pyplot.sca(axes[i])
        else:
            matplotlib.pyplot.sca(axes[i / ncols][i % ncols])

        matplotlib.pyplot.xlabel("$t$")
        matplotlib.pyplot.ylabel("x")
        matplotlib.pyplot.title((r"{\tt gpop_" + str(id) + "}").replace("_", r"\_"))

        x, y = numpy.meshgrid(density["time"].values, grid["x"].values)
        matplotlib.pyplot.pcolormesh(
            x, y, density.transpose().values[1:],
            cmap="CMRmap"
        )

    return container
