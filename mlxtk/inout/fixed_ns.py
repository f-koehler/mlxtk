from pathlib import Path
from typing import Tuple, Union

import h5py
import numpy
import pandas

from ..util import make_path


def read_fixed_ns_ascii(
        path: Union[str, Path]
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    path = make_path(path)

    with open(path, "r") as fptr:
        num_coefficients = (len(fptr.readline().split()) - 1) // 3

        names = ["time"] + [
            "real_" + str(i) for i in range(num_coefficients)
        ] + ["imag_" + str(i) for i in range(num_coefficients)]
        usecols = [i for i in range(2 * num_coefficients + 1)]
        data = pandas.read_csv(path,
                               delim_whitespace=True,
                               header=None,
                               names=names,
                               usecols=usecols)[names].values
        times, indices = numpy.unique(data[:, 0], return_index=True)
        num_times = len(times)

        return times, data[indices, 1:num_coefficients +
                           1], data[indices, num_coefficients + 1:]


def write_fixed_ns_hdf5(path: Union[str, Path], times: numpy.ndarray,
                        real: numpy.ndarray, imag: numpy.ndarray, N: int,
                        m: int):
    path = str(path)
    with h5py.File(path, "w") as fptr:
        grp = fptr.create_group("fixed_ns")
        grp.attrs["N"] = N
        grp.attrs["m"] = m

        dset = grp.create_dataset("time", times.shape, dtype=numpy.float64)
        dset[:] = times

        dset = grp.create_dataset("real", real.shape, dtype=numpy.float64)
        dset[:, :] = real

        dset = grp.create_dataset("imag", imag.shape, dtype=numpy.float64)
        dset[:, :] = imag