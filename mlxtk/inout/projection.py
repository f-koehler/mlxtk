import pandas
import numpy
import os

from . import hdf5
from ..log import get_logger


def read_basis_projection(path):
    parsed = hdf5.parse_hdf5_path(path)
    if parsed is None:
        return read_basis_projection_ascii(path)
    return read_basis_projection_hdf5(path)


def read_basis_projection_ascii(path):
    with open(path) as fhandle:
        num_coeffs = (len(fhandle.readline().split()) - 1) // 3

    usecols = ([0] + list(range(1, num_coeffs + 1)) +
               list(range(num_coeffs + 1, 2 * num_coeffs + 1)))
    names = (["time"] + ["real_" + str(i) for i in range(num_coeffs)] +
             ["imag_" + str(i) for i in range(num_coeffs)])
    return pandas.read_csv(
        path, delim_whitespace=True, usecols=usecols, names=names)


def read_basis_projection_hdf5(path):
    raise NotImplementedError


def add_projection_to_hdf5(group, path, name=None):
    logger = get_logger(__name__)

    data = read_basis_projection(path)

    if name is None:
        name = os.path.basename(path)

    opened_file = isinstance(group, str)
    if opened_file:
        logger.info("open new HDF5 file %s", group)
        group = h5py.File(group, "w")
    else:
        group = group.create_group(name)

    logger.info("add times")
    dset_time = group.create_dataset(
        "time", (len(data["time"]), ), dtype=numpy.float64, compression="gzip")
    dset_time[:] = data["time"]

    number_times = data.shape[0]
    number_coeff = (data.shape[1] - 1) // 2
    real_indices = ["real_" + str(i) for i in range(number_coeff)]
    imaginary_indices = ["imag_" + str(i) for i in range(number_coeff)]

    logger.info("add real parts")
    dset_real = group.create_dataset(
        "real", (number_times, number_coeff),
        dtype=numpy.float64,
        compression="gzip")
    dset_real[:, :] = data[real_indices]

    logger.info("add imaginary parts")
    dset_real = group.create_dataset(
        "imag_", (number_times, number_coeff),
        dtype=numpy.float64,
        compression="gzip")
    dset_real[:, :] = data[imaginary_indices]

    if opened_file:
        group.close()
