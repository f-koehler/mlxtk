from typing import Tuple

import numpy
import pandas

from . import tools


def read_dmat2_gridrep(
        path: str
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    is_hdf5, path, interior_path = tools.is_hdf5_path(path)
    if is_hdf5:
        assert interior_path
        raise NotImplementedError()

    return read_dmat2_gridrep_ascii(path)


def read_dmat2_gridrep_ascii(
        path: str
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    df = pandas.read_csv(path,
                         header=None,
                         names=["time", "x1", "x2", "dmat2"],
                         delim_whitespace=True)

    time = numpy.unique(df["time"].values)
    x1 = numpy.unique(df["x1"].values)
    x2 = numpy.unique(df["x2"].values)
    dmat2 = numpy.reshape(df["dmat2"].values, (len(time), len(x1), len(x2)))
    return time, x1, x2, dmat2
