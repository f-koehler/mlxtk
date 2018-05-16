from .signal_diff import compute_absolute_signal_diff_1d, compute_relative_signal_diff_1d


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


def compute_relative_natpop_diff(data1, data2, threshold=1e-5):
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
        time, diff = compute_relative_signal_diff_1d(
            time1, time2, data1[name].values / 1000.,
            data2[name].values / 1000., threshold)
        diffs.append(diff)

    return time, diffs
