import io
import re

import h5py
import numpy
import pandas


def read_natpop_ascii(path):
    re_timestamp = re.compile(r"^#time:\s+(.+)\s+\[au\]$")
    re_weight_info = re.compile(r"^Natural\s+weights")
    re_node_info = re.compile(r"^node:\s+(\d+)\s+layer:\s+(\d+)$")
    re_orbitals_start = re.compile(r"^m(\d+):\s+(.+)$")

    # read whole file
    with open(path) as fh:
        content = fh.readlines()

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
        m = re_timestamp.match(line)
        if m:
            timestamps.append(float(m.group(1)))
            continue

        # check for "node: x    layer: y" line
        m = re_node_info.match(line)
        if m:
            current_node = int(m.group(1)) - 1
            if current_node not in node_content:
                node_content[current_node] = {}
            continue

        # check for "mx: xxx xxx xxx ... " line
        m = re_orbitals_start.match(line)
        if m:
            current_orbitals = int(m.group(1)) - 1
            if current_orbitals not in node_content[current_node]:
                node_content[current_node][current_orbitals] = []
            node_content[current_node][current_orbitals].append(m.group(2))
            continue

        # found continued data line
        node_content[current_node][current_orbitals][-1] += " " + line

    # create DataFrames
    data = {}
    for node in node_content:
        data[node + 1] = {}
        for orbitals in node_content[node]:
            # obtain number of orbitals
            num_orbitals = len(node_content[node][orbitals][0].split())

            # prepend time stamps to data
            for i in range(len(timestamps)):
                node_content[node][orbitals][i] = node_content[node][orbitals][i]

            # construct header for DataFrame
            header = (
                " ".join(
                    ["orbital_" + str(orbital) for orbital in range(0, num_orbitals)]
                )
                + "\n"
            )

            # create DataFrame
            sio = io.StringIO(header + "\n".join(node_content[node][orbitals]))
            df = pandas.read_csv(sio, sep=r"\s+")
            vals = numpy.zeros(df.shape, dtype=numpy.float64)
            for i in range(num_orbitals):
                vals[:, i] = df["orbital_" + str(i)]
            data[node + 1][orbitals + 1] = vals / 1000.

    return [numpy.array(timestamps), data]


def read_natpop_hdf5(path, node=None, dof=None):
    with h5py.File(path, "r") as fp:
        time = fp["time"][:]

        if node is not None:
            if dof is not None:
                return [time, fp["node_" + str(node)]["dof_" + str(dof)][:, :]]
            else:
                node_str = "node_" + str(node)
                data = {}
                for entry in fp[node_str]:
                    data[int(entry.replace("dof_", ""))] = fp[node_str][entry][:, :]
                return [time, data]
        else:
            if dof is not None:
                raise ValueError("No node specified while specifying the DOF")
            data = {}
            for node_entry in (entry for entry in fp if entry.startswith("node_")):
                node_i = int(node_entry.replace("node_", ""))
                data[node_i] = {}
                for dof_entry in fp[node_entry]:
                    dof_i = int(dof_entry.replace("dof_", ""))
                    data[node_i][dof_i] = fp[node_entry][dof_entry]
            return [time, data]


def write_natpop_hdf5(path, data):
    time, data = data
    with h5py.File(path, "w") as fp:
        dset = fp.create_dataset(
            "time", time.shape, dtype=numpy.float64, compression="gzip"
        )
        dset[:] = time
        for node in data:
            grp = fp.create_group("node_{}".format(node))
            for dof in data[node]:
                dset = grp.create_dataset(
                    "dof_" + str(dof),
                    data[node][dof].shape,
                    dtype=numpy.float64,
                    compression="gzip",
                )
                dset[:, :] = data[node][dof][:, :]
