import numpy
import scipy.interpolate

from .. import log
from ..inout.expval import read_expval


def get_label_expval(operator):
    return r"{\left<" + operator + r"\right>(t)}"


def get_label_real_part(label):
    return r"{\mathrm{Re}\left[" + label + r"\right]}"


def get_label_imaginary_part(label):
    return r"{\mathrm{Im}\left[" + label + r"\right]}"


def plot_expval(plot, path, real=True, imaginary=False):
    data = read_expval(path)

    lines = []

    if real:
        label = "$" + get_label_real_part(get_label_expval("O")) + "$"
        lines += plot.axes.plot(data.time, data.real, color="C0", label=label)
        plot.axes.set_ylabel(label)

    if imaginary:
        label = "$" + get_label_imaginary_part(get_label_expval("O")) + "$"
        ax = plot.axes.twinx() if real else plot.axes
        lines += ax.plot(data.time, data.imaginary, color="C1", label=label)
        ax.set_ylabel(label)

    plot.axes.set_xlabel("$t$")
    if lines:
        plot.axes.legend(lines, [l.get_label() for l in lines])


def plot_variance(*args, **kwargs):
    return plot_expval(*args, **kwargs)


def plot_expval_diff(plot,
                     path1,
                     path2,
                     real=True,
                     imaginary=False,
                     relative=False):
    data1 = read_expval(path1)
    data2 = read_expval(path2)

    t_min = max(min(data1.time), min(data2.time))
    t_max = min(max(data1.time), max(data2.time))
    n_t = max(len(data1.time), len(data2.time))

    interpolate = numpy.array_equal(data1.time.as_matrix(),
                                    data2.time.as_matrix())

    lines = []
    if not interpolate:
        if real:
            if relative:
                label = r"$1-\frac{" + get_label_real_part(
                    get_label_expval("O") + "_2") + "}{" + get_label_real_part(
                        get_label_expval("O") + "_1") + "}$"
                lines += plot.axes.plot(
                    data1.time,
                    1. - data2.real / data1.real,
                    color="C0",
                    label=label)
            else:
                label = "$" + get_label_real_part(
                    get_label_expval("O") + "_1") + " -" + get_label_real_part(
                        get_label_expval("O") + "_2") + "$"
                lines += plot.axes.plot(
                    data1.time,
                    data1.real - data2.real,
                    color="C0",
                    label=label)

            plot.axes.set_ylabel(label)

        if imaginary:
            ax = plot.axes.twinx() if real else plot.axes
            if relative:
                label = r"$1-\frac{" + get_label_imaginary_part(
                    get_label_expval("O") + "_2"
                ) + "}{" + get_label_imaginary_part(
                    get_label_expval("O") + "_1") + "}$"
                lines += ax.plot(
                    data1.time,
                    1. - data2.imaginary / data1.imaginary,
                    color="C0",
                    label=label)
            else:
                label = "$" + get_label_imaginary_part(
                    get_label_expval("O") + "_1"
                ) + " -" + get_label_imaginary_part(
                    get_label_expval("O") + "_2") + "$"
                lines += ax.plot(
                    data1.time,
                    data1.imaginary - data2.imaginary,
                    color="C0",
                    label=label)

            ax.set_ylabel(label)
    else:
        log.get_logger(__name__).warn("incompatible times, interpolating")
        t = numpy.linspace(t_min, t_max, n_t)

        if real:
            interp1 = scipy.interpolate.interp1d(
                data1.time,
                data1.real,
                kind=5,
                bounds_error=True,
                assume_sorted=True)
            interp2 = scipy.interpolate.interp1d(
                data2.time,
                data2.real,
                kind=5,
                bounds_error=True,
                assume_sorted=True)

            if relative:
                label = r"$1-\frac{" + get_label_real_part(
                    get_label_expval("O") + "_2") + "}{" + get_label_real_part(
                        get_label_expval("O") + "_1") + "}$"
                lines += plot.axes.plot(
                    t, 1. - interp2(t) / interp1(t), color="C0", label=label)
            else:
                label = "$" + get_label_real_part(
                    get_label_expval("O") + "_1") + " -" + get_label_real_part(
                        get_label_expval("O") + "_2") + "$"
                lines += plot.axes.plot(
                    t, interp1(t) - interp2(t), color="C0", label=label)

            plot.axes.set_ylabel(label + " (interpolated)")

        if imaginary:
            ax = plot.axes.twinx() if real else plot.axes
            interp1 = scipy.interpolate.interp1d(
                data1.time,
                data1.imaginary,
                kind=5,
                bounds_error=True,
                assume_sorted=True)
            interp2 = scipy.interpolate.interp1d(
                data2.time,
                data2.imaginary,
                kind=5,
                bounds_error=True,
                assume_sorted=True)

            if relative:
                label = r"$1-\frac{" + get_label_imaginary_part(
                    get_label_expval("O") + "_2"
                ) + "}{" + get_label_imaginary_part(
                    get_label_expval("O") + "_1") + "}$"
                lines += ax.plot(
                    t, 1. - interp2(t) / interp1(t), color="C0", label=label)
            else:
                label = "$" + get_label_imaginary_part(
                    get_label_expval("O") + "_1"
                ) + " -" + get_label_imaginary_part(
                    get_label_expval("O") + "_2") + "$"
                lines += ax.plot(
                    t, interp1(t) - interp2(t), color="C0", label=label)

            plot.axes.set_ylabel(label + " (interpolated)")


def plot_variance_diff(*args, **kwargs):
    plot_expval_diff(*args, **kwargs)
