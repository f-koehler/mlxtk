import numpy
import pandas


def read_norbs_ascii(path):
    data = pandas.read_csv(
        path, header=None, delim_whitespace=True).values.transpose()
    grid = numpy.unique(data[1])
    times = numpy.unique(data[0])
    m = (data.shape[0] - 2) // 2
    orb_size = len(grid)

    converted = []
    for i, _ in enumerate(times):
        converted.append([])
        for j in range(m):
            converted[-1].append(
                pandas.DataFrame({
                    "real":
                    data[2 + j * 2][i * orb_size:(i + 1) * orb_size],
                    "imaginary":
                    data[2 + j * 2 + 1][i * orb_size:(i + 1) * orb_size],
                }))
    return times, grid, converted
