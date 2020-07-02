import io
import re
from pathlib import Path
from typing import Dict, Tuple, Union

import h5py
import numpy
import pandas

from mlxtk.inout import tools


def read_natpop(
    path: str, node: int = 0, dof: int = 0
) -> Tuple[
    numpy.ndarray, Union[Dict[int, numpy.ndarray], Dict[int, Dict[int, numpy.ndarray]]]
]:
    is_hdf5, path, interior_path = tools.is_hdf5_path(path)
    if is_hdf5:
        return read_natpop_hdf5(path, interior_path, node, dof)

    return read_natpop_ascii(path, node, dof)


def read_natpop_ascii(
    path: str, node: int = 0, dof: int = 0
) -> Tuple[
    numpy.ndarray, Union[Dict[int, numpy.ndarray], Dict[int, Dict[int, numpy.ndarray]]]
]:
    re_timestamp = re.compile(r"^#time:\s+(.+)\s+\[au\]$")
    re_weight_info = re.compile(r"^Natural\s+weights")
    re_node_info = re.compile(r"^node:\s+(\d+)\s+layer:\s+(\d+)$")
    re_orbitals_start = re.compile(r"^m(\d+):\s+(.+)$")

    # read whole file
    with open(path) as fptr:
        content = fptr.readlines()

    timestamps = []
    node_content = {}

    current_node = None
    current_orbitals = None

    for line in content:
        line = line.strip()

        # skip empty lines
        if not line:
            continue

        # skip useless info lines
        if re_weight_info.match(line):
            continue

        # gather timestamps
        match = re_timestamp.match(line)
        if match:
            timestamps.append(float(match.group(1)))
            continue

        # check for "node: x    layer: y" line
        match = re_node_info.match(line)
        if match:
            current_node = int(match.group(1)) - 1
            if current_node not in node_content:
                node_content[current_node] = {}
            continue

        # check for "mx: xxx xxx xxx ... " line
        match = re_orbitals_start.match(line)
        if match:
            current_orbitals = int(match.group(1)) - 1
            if current_orbitals not in node_content[current_node]:
                node_content[current_node][current_orbitals] = []
            node_content[current_node][current_orbitals].append(match.group(2))
            continue

        # found continued data line
        node_content[current_node][current_orbitals][-1] += " " + line

    # create DataFrames
    data = {}
    for n in node_content:
        data[n + 1] = {}
        for orbitals in node_content[n]:
            # obtain number of orbitals
            num_orbitals = len(node_content[n][orbitals][0].split())

            # construct header for DataFrame
            header = (
                " ".join(
                    ["orbital_" + str(orbital) for orbital in range(0, num_orbitals)]
                )
                + "\n"
            )

            # create DataFrame
            sio = io.StringIO(header + "\n".join(node_content[n][orbitals]))
            dataFrame = pandas.read_csv(sio, sep=r"\s+")
            vals = numpy.zeros(dataFrame.shape, dtype=numpy.float64)
            for i in range(num_orbitals):
                vals[:, i] = dataFrame["orbital_" + str(i)]
            data[n + 1][orbitals + 1] = vals / 1000.0

    if node:
        if dof:
            return (numpy.array(timestamps), data[node][dof])
        return (numpy.array(timestamps), data[node])

    return (numpy.array(timestamps), data)


def read_natpop_hdf5(
    path: Union[str, Path], interior_path: str, node: int = None, dof: int = None
) -> Union[
    Tuple[numpy.ndarray, numpy.ndarray],
    Tuple[numpy.ndarray, Dict[str, Dict[str, numpy.ndarray]]],
]:
    with h5py.File(path, "r") as fptr:
        time = fptr[interior_path + "/time"][:]

        if node and dof:
            data = fptr[interior_path + "/node_{}/dof_{}".format(node, dof)][:, :]
            return (time, data)

        data = {}
        for node_str in fptr[interior_path]:
            data[node_str] = {}
            for dof_str in fptr[interior_path + "/" + node_str]:
                data[node_str][dof_str] = fptr[
                    interior_path + "/" + node_str + "/" + dof_str
                ][:, :]
        return data


def add_natpop_to_hdf5(
    fptr: Union[h5py.File, h5py.Group],
    time: numpy.ndarray,
    natpops: Dict[int, Dict[int, numpy.ndarray]],
):
    fptr.create_dataset("time", time.shape, dtype=numpy.float64)[:] = time

    for node in natpops:
        grp = fptr.create_group("node_" + str(node))
        for dof in natpops[node]:
            grp.create_dataset(
                "dof_" + str(dof), natpops[node][dof].shape, dtype=numpy.float64
            )[:, :] = natpops[node][dof]
