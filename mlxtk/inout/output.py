from pathlib import Path
from typing import Tuple, Union

import h5py
import numpy
import pandas

from mlxtk.inout import tools


def read_output(
    path: Union[Path, str]
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    is_hdf5, path, interior_path = tools.is_hdf5_path(path)
    if is_hdf5:
        return read_output_hdf5(path, interior_path)

    return read_output_ascii(path)


def read_output_ascii(
    path: Union[Path, str]
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Read an output file (raw ASCII format)

    Args:
       path (str): path to the file

    Return:
       list: The contents of the output file as a :py:class:`list` containing
          four :py:class:`numpy.ndarray` instances. The first one contains all
          simulation times. The other entries contain the norm, energy and
          maximum SPF overlap of the wave function at all times.
    """
    dataFrame = pandas.read_csv(
        str(path), sep=r"\s+", names=["time", "norm", "energy", "overlap"]
    )
    return (
        dataFrame["time"].values,
        dataFrame["norm"].values,
        dataFrame["energy"].values,
        dataFrame["overlap"].values,
    )


def read_output_hdf5(
    path: Union[Path, str], interior_path: str = "/"
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    with h5py.File(path, "r") as fptr:
        return (
            fptr[interior_path]["time"][:],
            fptr[interior_path]["norm"][:],
            fptr[interior_path]["energy"][:],
            fptr[interior_path]["overlap"][:],
        )


def add_output_to_hdf5(
    fptr: Union[h5py.File, h5py.Group],
    time: numpy.ndarray,
    norm: numpy.ndarray,
    energy: numpy.ndarray,
    overlap: numpy.ndarray,
):
    fptr.create_dataset("time", time.shape, dtype=numpy.float64)[:] = time
    fptr.create_dataset("norm", norm.shape, dtype=numpy.float64)[:] = norm
    fptr.create_dataset("energy", energy.shape, dtype=numpy.float64)[:] = energy
    fptr.create_dataset("overlap", overlap.shape, dtype=numpy.float64)[:] = overlap
