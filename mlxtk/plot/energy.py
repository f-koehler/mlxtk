import numpy
import scipy.interpolate

from ..inout.output import read_output
from ..tools.signal_diff import compute_absolute_signal_diff_1d, compute_relative_signal_diff_1d


def plot_energy_diff(plot, path1, path2, relative=False):
    data1 = read_output(path1)
    data2 = read_output(path2)

    t_min = max(min(data1.time), min(data2.time))
    t_max = min(max(data1.time), max(data2.time))
    n_t = max(len(data1.time), len(data2.time))

    if relative:
        label = r"$1-\frac{{\langle H\rangle}_2 (t)}{{\langle H\rangle}_1 (t)}$"
        t, values = compute_relative_signal_diff_1d(data1.time, data2.time,
                                                    data1.energy, data2.energy)
    else:
        label = r"${\langle H\rangle}_1 (t)-{\langle H\rangle}_2 (t)$"
        t, values = compute_absolute_signal_diff_1d(data1.time, data2.time,
                                                    data1.energy, data2.energy)

    plot.axes.set_xlabel("$t$")
    plot.axes.set_ylabel(label)
    plot.axes.plot(t, values)
