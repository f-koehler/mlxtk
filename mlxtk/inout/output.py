import h5py
import numpy
import pandas
import os.path

from mlxtk import log
from mlxtk.inout import hdf5


def read_output(path):
    """Read a QDTK ouput file

    The resulting :py:class:`pandas.DataFrame` contains the columns `time`,
    `norm`, `energy` and `overlap`.

    Args:
        path (str): Path to the output file

    Returns:
        pandas.DataFrame: The output of the QDTK run
    """
    parsed = hdf5.parse_hdf5_path(path)
    if parsed is None:
        return read_output_ascii(path)
    else:
        return read_output_hdf5(parsed)


def read_output_ascii(path):
    return pandas.read_csv(
        path, sep=r"\s+", names=["time", "norm", "energy", "overlap"])


def read_output_hdf5(parsed_path):
    path, path_inside = parsed_path
    if not hdf5.is_hdf5_group(path, path_inside):
        raise RuntimeError("Expected a group containing the output")

    with h5py.File(path, "r") as fhandle:
        data = pandas.DataFrame(
            data={
                "time": fhandle[os.path.join(path_inside, "time")][:],
                "norm": fhandle[os.path.join(path_inside, "norm")][:],
                "energy": fhandle[os.path.join(path_inside, "energy")][:],
                "overlap": fhandle[os.path.join(path_inside, "overlap")][:],
            })

    return data


def add_output_to_hdf5(group, output_path):
    logger = log.get_logger(__name__)

    data = read_output(output_path)

    opened_file = isinstance(group, str)
    if opened_file:
        logger.info("open new hdf5 file %s", group)
        group = h5py.File(group, "w")
    else:
        group = group.create_group("output")

    logger.info("add times")
    dset_time = group.create_dataset(
        "time", (len(data["time"]), ), dtype=numpy.float64, compression="gzip")
    dset_time[:] = data["time"]

    logger.info("add norm")
    dset_norm = group.create_dataset(
        "norm", (len(data["norm"]), ), dtype=numpy.float64, compression="gzip")
    dset_norm[:] = data["norm"]

    logger.info("add energy")
    dset_energy = group.create_dataset(
        "energy", (len(data["energy"]), ),
        dtype=numpy.float64,
        compression="gzip")
    dset_energy[:] = data["energy"]

    logger.info("add overlap")
    dset_overlap = group.create_dataset(
        "overlap", (len(data["overlap"]), ),
        dtype=numpy.float64,
        compression="gzip")
    dset_overlap[:] = data["overlap"]

    if opened_file:
        group.close()
