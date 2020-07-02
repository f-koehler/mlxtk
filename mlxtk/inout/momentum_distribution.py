from pathlib import Path
from typing import Tuple, Union

import h5py
import numpy
import pandas


def read_momentum_distribution_ascii(
    path: Union[str, Path]
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    data = pandas.read_csv(path, sep=r"\s+", names=["times", "momenta", "distribution"])
    times = numpy.sort(numpy.unique(data["times"].to_numpy()))
    num_times = len(times)
    num_momenta = len(data["momenta"]) // num_times
    momenta = data["momenta"].to_numpy()[:num_momenta]
    distribution = numpy.reshape(
        data["distribution"].to_numpy(), (num_times, num_momenta)
    )
    return times, momenta, distribution


def read_momentum_distribution_hdf5(
    path: Union[Path, str], interior_path: str
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    with h5py.File(path, "r") as fptr:
        return (
            fptr[interior_path]["times"][:],
            fptr[interior_path]["momenta"][:],
            fptr[interior_path]["distribution"][:, :],
        )


# def read_natural_orbitals_momentum_representation(
#     path: Union[str,
#                 Path]) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
#     with open(path, "r") as fptr:
#         num_orbitals = len(fptr.readline().split()) - 2

#     names = ["times", "momenta"
#              ] + ["orbital_" + str(i) for i in range(num_orbitals)]
#     data = pandas.read_csv(path, sep=r"\s+", names=names)
#     times = numpy.sort(numpy.unique(data["times"].to_numpy()))
#     num_times = len(times)
#     num_momenta = len(data["momenta"]) // num_times
#     momenta = data["momenta"].to_numpy()[:num_momenta]


def add_momentum_distribution_to_hdf5(
    fptr: Union[h5py.File, h5py.Group],
    times: numpy.ndarray,
    momenta: numpy.ndarray,
    distribution: numpy.ndarray,
):
    grp = fptr.create_group("momentum_distribution")
    grp.create_dataset("times", shape=times.shape, dtype=times.dtype)[:] = times
    grp.create_dataset("momenta", shape=momenta.shape, dtype=momenta.dtype)[:] = momenta
    grp.create_dataset(
        "distribution", shape=distribution.shape, dtype=distribution.dtype
    )[:, :] = distribution
