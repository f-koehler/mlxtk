"""Read one-body density from ASCII and HDF5 files.
"""
import io
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy
import pandas

from mlxtk.inout import tools

assert List


def read_gpop(
    path: str, dof: Optional[int] = None
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Read the one-body densities from a file.

    This method detects whether the file is an ASCII or a HDF5 file.

    Args:
        path (str): Path to the file to read
        dof (int): Only read data for desired degree of freedom (optional)

    Returns:
        One-body density data
    """
    is_hdf5, path, interior_path = tools.is_hdf5_path(path)
    if is_hdf5:
        return read_gpop_hdf5(path, interior_path, dof)

    return read_gpop_ascii(path, dof)


def read_gpop_ascii(
    path: str, dof: Optional[int] = None
) -> Union[
    Tuple[numpy.ndarray, numpy.ndarray],
    Tuple[numpy.ndarray, Dict[int, numpy.ndarray], Dict[int, numpy.ndarray]],
]:
    """Read the one-body densities from a raw ML-X file.

    Args:
        path (str): path of the ASCII file

    Returns:
        Tuple[numpy.ndarray, Dict[int, numpy.ndarray], Dict[int, numpy.ndarray]
        ]: one-body density data
    """
    regex_time_stamp = re.compile(r"^#\s+(.+)\s+\[au\]$")
    times = []
    dofs = []  # type: List[int]
    grids = {}
    densities = {}

    with open(path, "r") as fhandle:
        match = regex_time_stamp.match(fhandle.readline())
        if match:
            times.append(float(match.group(1)))

        while True:
            try:
                dof_s, grid_points_s = fhandle.readline().split()
            except ValueError:
                raise RuntimeError("Failed to determine DOF and number of grid points")
            dof_index = int(dof_s)
            grid_points = int(grid_points_s)

            sio = io.StringIO()
            for _ in range(0, grid_points):
                sio.write(fhandle.readline())
            sio.seek(0)
            data = pandas.read_csv(sio, sep=r"\s+", names=["grid", "density"])
            del sio

            if dof_index not in dofs:
                dofs.append(dof_index)
                grids[dof_index] = data["grid"].values
                densities[dof_index] = [data["density"].values]
            else:
                densities[dof_index].append(data["density"])

            # read the empty line after the block
            fhandle.readline()

            # check next line
            position = fhandle.tell()
            line = fhandle.readline()
            if not line:
                # end of file reached
                break

            match = regex_time_stamp.match(line)
            if match:
                # new time stamp
                times.append(float(match.group(1)))
                continue

            # new dof
            fhandle.seek(position)

    # convert densities to numpy arrays
    for dof_index in densities:
        densities[dof_index] = numpy.array(densities[dof_index])

    if dof is None:
        return (numpy.array(times), grids, densities)

    return (numpy.array(times), grids[dof], densities[dof])


def read_gpop_hdf5(
    path: Union[Path, str], interior_path: str, dof: Optional[int] = None
) -> Union[
    Tuple[numpy.ndarray, Dict[int, numpy.ndarray], Dict[int, numpy.ndarray]],
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray],
]:
    """Read the one-body densities from a HDF5 file.

    Args:
        path (str): path of the HDF5 file

    Returns:
        one-body density data
    """
    with h5py.File(path, "r") as fp:
        time = fp[interior_path]["time"][:]
        if dof is not None:
            dof_str = "dof_" + str(dof)
            return (
                time,
                fp[interior_path][dof_str]["grid"][:],
                fp[interior_path][dof_str]["density"][:, :],
            )

        grids = {}
        densities = {}
        for dof_str in (
            key for key in fp[interior_path].keys() if key.startswith("dof_")
        ):
            dof_i = int(dof_str.replace("dof_", ""))
            grids[dof_i] = fp[interior_path][dof_str]["grid"][:]
            densities[dof_i] = fp[interior_path][dof_str]["density"][:, :]
        return (time, grids, densities)


def add_gpop_to_hdf5(
    fptr: Union[h5py.Group, h5py.File],
    time: numpy.ndarray,
    grids: Dict[int, numpy.ndarray],
    densities: Dict[int, numpy.ndarray],
):
    fptr.create_dataset("time", time.shape, dtype=numpy.float64)[:] = time

    for dof in grids:
        grp = fptr.create_group("dof_" + str(dof))

        grp.create_dataset("grid", grids[dof].shape, dtype=numpy.float64)[:] = grids[
            dof
        ]

        grp.create_dataset("density", densities[dof].shape, dtype=numpy.float64)[
            :, :
        ] = densities[dof]
