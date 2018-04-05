from ..inout.natpop import read_natpop


def plot_natpop(plot, path):
    data = read_natpop(path)[1][1]
    for column in data.columns[1:]:
        plot.axes.plot(data["time"], data[column] / 1000.)
    plot.axes.set_xlabel("$t$")
    plot.axes.set_ylabel(r"$\lambda_i$")
