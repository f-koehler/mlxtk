from .signal_diff import compute_absolute_signal_diff_1d, compute_relative_signal_diff_1d
import numpy


def compute_absolute_natpop_diff(data1, data2):
    time1 = data1["time"].values
    time2 = data2["time"].values

    norbs = data1.shape[1] - 1
    norbs2 = data1.shape[1] - 1

    if norbs != norbs2:
        raise RuntimeError(
            "Different number of orbitals, cannot calculate difference")

    diffs = []
    for i in range(norbs):
        name = "orbital" + str(i)
        time, diff = compute_absolute_signal_diff_1d(
            time1, time2, data1[name].values / 1000.,
            data2[name].values / 1000.)
        diffs.append(diff)

    return time, diffs


def compute_relative_natpop_diff(data1, data2, threshold=1e-2):
    time1 = data1["time"].values
    time2 = data2["time"].values

    norbs = data1.shape[1] - 1
    norbs2 = data1.shape[1] - 1

    if norbs != norbs2:
        raise RuntimeError(
            "Different number of orbitals, cannot calculate difference")

    time = None
    length = None
    diffs = []
    for i in range(norbs):
        name = "orbital" + str(i)
        natpop1 = data1[name].values / 1000.
        natpop2 = data2[name].values / 1000.

        if numpy.max(numpy.abs(natpop1)) < 1e-2 and numpy.max(
                numpy.abs(natpop2)) < 1e-2:
            diffs.append([])
            continue

        time, diff = compute_relative_signal_diff_1d(time1, time2, natpop1,
                                                     natpop2, threshold)
        length = len(time)
        diffs.append(diff)

    for i in range(len(diffs)):
        if diffs[i] == []:
            diffs[i] = numpy.zeros(length)

    return time, diffs
