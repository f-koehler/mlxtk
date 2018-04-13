import numpy
import scipy.interpolate

from .. import log
from ..inout.output import read_output


def plot_energy_diff(plot, path1, path2, relative=False):
    data1 = read_output(path1)
    data2 = read_output(path2)

    t_min = max(min(data1.time), min(data2.time))
    t_max = min(max(data1.time), max(data2.time))
    n_t = max(len(data1.time), len(data2.time))

    interpolate = numpy.array_equal(data1.time.as_matrix(),
                                    data2.time.as_matrix())

    if not interpolate:
        if relative:
            plot.axes.plot(
                data1.time, 1 - data2.energy / data1.energy, marker=".")
            plot.axes.set_ylabel(
                r"$1-\frac{{\langle H\rangle}_2 (t)}{{\langle H\rangle}_1 (t)}$"
            )
        else:
            plot.axes.plot(data1.time, data1.energy - data2.energy, marker=".")
            plot.axes.set_ylabel(
                r"${\langle H\rangle}_1 (t)-{\langle H\rangle}_2 (t)$")
        plot.axes.set_xlabel("$t$")
    else:
        log.get_logger(__name__).warn("incompatible times, interpolating")
        t = numpy.linspace(t_min, t_max, n_t)
        interp1 = scipy.interpolate.interp1d(
            data1.time,
            data1.energy,
            kind=5,
            bounds_error=True,
            assume_sorted=True)
        interp2 = scipy.interpolate.interp1d(
            data2.time,
            data2.energy,
            kind=5,
            bounds_error=True,
            assume_sorted=True)
        if relative:
            plot.axes.set_ylabel(
                r"$1-\frac{{\langle H\rangle}_2 (t)}{{\langle H\rangle}_1 (t)}$ (interpolated)"
            )
            plot.axes.plot(t, 1 - interp2(t) / interp1(t), marker=".")
        else:
            plot.axes.set_ylabel(
                r"${\langle H\rangle}_1 (t)-{\langle H\rangle}_2 (t)$ (interpolated)"
            )
            plot.axes.plot(t, interp1(t) - interp2(t), marker=".")
        plot.axes.set_xlabel("$t$")
