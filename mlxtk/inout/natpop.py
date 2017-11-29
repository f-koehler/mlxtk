import re

import numpy
import pandas
import h5py

from mlxtk.stringio import StringIO
from mlxtk import log


def read_natpop(path):
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
            for i, time in enumerate(timestamps):
                node_content[node][orbitals][
                    i] = str(time) + " " + node_content[node][orbitals][i]

            # construct header for DataFrame
            header = "time " + " ".join([
                "orbital" + str(orbital) for orbital in range(0, num_orbitals)
            ]) + "\n"

            # create DataFrame
            sio = StringIO(header + "\n".join(node_content[node][orbitals]))
            data[node + 1][orbitals + 1] = pandas.read_csv(sio, sep="\s+")

    return data


def add_natpop_to_hdf5(group, natpop_path):
    logger = log.getLogger("h5py")

    opened_file = isinstance(group, str)
    if opened_file:
        logger.info("open new hdf5 file %s", group)
        group = h5py.File(group, "w")

    group_natpop = group.create_group("natpop")

    data = read_natpop(natpop_path)
    for node in data:
        group_node = group_natpop.create_group("node" + str(node))
        for layer in data[node]:
            logger.info("add natpop data (node: %d, layer: %d)", node, layer)
            current_data = data[node][layer]
            dataset_layer = group_node.create_dataset(
                "layer" + str(layer),
                (current_data.shape[0], current_data.shape[1] - 1),
                dtype=numpy.float64,
                compression="gzip")
            dataset_layer[:, :] = current_data.as_matrix()[:, 1:]

    if opened_file:
        logger.info("close hdf5 file")
        group.close()
