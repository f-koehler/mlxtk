import matplotlib.pyplot
import os
import re

import mlxtk.inout.natpop
import mlxtk.plot.container


def plot_overview(dir, ncols=1):
    re_file = re.compile(r"^natpop_(\d+)\.gz$")

    files = []
    ids = []
    for entry in os.listdir(dir):
        path = os.path.join(dir, entry)
        if not os.path.isfile(path):
            continue

        m = re_file.match(entry)
        if not m:
            continue

        files.append(path)
        ids.append(m.group(1))

    if not ids:
        return

    n = len(files)
    ncols = min(n, ncols)
    nrows = round(float(n) / ncols)

    fig, axes = matplotlib.pyplot.subplots(nrows=nrows, ncols=ncols)
    container = mlxtk.plot.container.PlotContainer(fig, axes)

    for i, id in enumerate(ids):
        data = mlxtk.inout.natpop.read(dir, id)

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
        matplotlib.pyplot.ylabel("natural population")
        matplotlib.pyplot.xlim(data["time"][0], data["time"].values[-1])
        matplotlib.pyplot.title(
            (r"{\tt natpop_" + id + "}").replace("_", r"\_"))

        for col in data:
            if col == "time":
                continue
            matplotlib.pyplot.plot(data["time"], data[col])

    return container
