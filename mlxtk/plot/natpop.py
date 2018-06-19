from ..inout.natpop import read_natpop
from ..tools.natpop_diff import (
    compute_absolute_natpop_diff,
    compute_relative_natpop_diff,
)


def plot_natpop(plot, path):
    data = read_natpop(path)[1][1]
    for column in data.columns[1:]:
        plot.axes.plot(data["time"], data[column] / 1000.)
    plot.axes.set_xlabel("$t$")
    plot.axes.set_ylabel(r"$\lambda_i$")


def plot_natpop_diff(plot, path1, path2, relative=False):
    data1 = read_natpop(path1)[1][1]
    data2 = read_natpop(path2)[1][1]

    if relative:
        time, diffs = compute_relative_natpop_diff(data1, data2)
        label = r"$1-\frac{\lambda_i^{(2)}}{\lambda_i^{(1)}}$"
    else:
        time, diffs = compute_absolute_natpop_diff(data1, data2)
        label = r"$\lambda_i^{(1)}-\lambda_i^{(2)}$"

    for diff in diffs:
        plot.axes.plot(time, diff)

    plot.axes.set_xlabel("$t$")
    plot.axes.set_ylabel(label)
