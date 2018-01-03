import h5py
import numpy
import os.path
import pandas

from mlxtk import log
from . import hdf5


def read_expval(path):
    parsed = hdf5.parse_hdf5_path(path)
    if parsed is None:
        return read_expval_ascii(path)
    else:
        return read_expval_hdf5(parsed)


def read_expval_ascii(path):
    data = pandas.read_csv(
        path, sep=r"\s+", names=["time", "real", "imaginary"])

    return data


def read_expval_hdf5(parsed_path):
    path, path_inside = parsed_path
    if not hdf5.is_hdf5_group(path, path_inside):
        raise RuntimeError("Expected a group containing the output")

    with h5py.File(path, "r") as fhandle:
        data = pandas.DataFrame(
            data={
                "time": fhandle[os.path.join(path_inside, "time")][:],
                "real": fhandle[os.path.join(path_inside, "real")][:],
                "imaginary": fhandle[os.path.join(path_inside, "imaginary")][:]
            })

    return data


def add_expval_to_hdf5(group, expval_path):
    logger = log.get_logger(__name__)

    data = read_expval(expval_path)

    name = "expval_" + os.path.splitext(os.path.basename(expval_path))[0]

    opened_file = isinstance(group, str)
    if opened_file:
        logger.info("open new hdf5 file %s", group)
        group = h5py.File(group, "w")
    else:
        group = group.create_group(name)

    logger.info("add times")
    dset_time = group.create_dataset(
        "time", (len(data["time"]), ), dtype=numpy.float64, compression="gzip")
    dset_time[:] = data["time"]

    logger.info("add real part")
    dset_time = group.create_dataset(
        "real", (len(data["real"]), ), dtype=numpy.float64, compression="gzip")
    dset_time[:] = data["real"]

    logger.info("add real imaginary")
    dset_time = group.create_dataset(
        "imaginary", (len(data["imaginary"]), ),
        dtype=numpy.float64,
        compression="gzip")
    dset_time[:] = data["imaginary"]

    if opened_file:
        group.close()
