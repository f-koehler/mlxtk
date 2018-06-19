import h5py
import numpy
import os
import pandas
import re

from mlxtk import log
from mlxtk.stringio import StringIO
from mlxtk.inout import hdf5
from . import InOutError


def read_gpop(path):
    parsed = hdf5.parse_hdf5_path(path)
    if parsed is None:
        return read_gpop_ascii(path)
    else:
        return read_gpop_hdf5(parsed)


def read_gpop_ascii(path):
    regex_time_stamp = re.compile(r"^#\s+(.+)\s+\[au\]$")
    times = []
    dofs = []
    grids = {}
    densities = {}

    with open(path, "r") as fhandle:
        match = regex_time_stamp.match(fhandle.readline())
        if match:
            times.append(float(match.group(1)))

        while True:
            try:
                dof, grid_points = fhandle.readline().split()
            except ValueError:
                raise InOutError(
                    "Failed to determine DOF and number of grid points")
            dof = int(dof)
            grid_points = int(grid_points)

            sio = StringIO()
            for i in range(0, grid_points):
                sio.write(fhandle.readline())
            sio.seek(0)
            data = pandas.read_csv(sio, sep=r"\s+", names=["grid", "density"])
            sio = None

            # unfortuantely reading from the fhandle directly does not seem to
            # work
            # data = pandas.read_csv(
            #     fhandle,
            #     sep=r"\s+",
            #     nrows=grid_points,
            #     names=["grid", "density"],
            #     skip_blank_lines=False)

            if dof not in dofs:
                dofs.append(dof)
                grids[dof] = data["grid"].values
                densities[dof] = [data["density"].values]
            else:
                densities[dof].append(data["density"])

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
    for dof in densities:
        densities[dof] = numpy.array(densities[dof])

    return numpy.array(times), grids, densities


def read_gpop_hdf5(parsed_path):
    path, path_inside = parsed_path
    if not hdf5.is_hdf5_group(path, path_inside):
        raise hdf5.HDF5Error("Expected a group containing the densities")

    fhandle = h5py.File(path, "r")

    times = fhandle[os.path.join(path_inside, "times")][:]

    regex_grid_name = re.compile(r"^grid(\d+)$")
    regex_density_name = re.compile(r"^density(\d+)$")
    group_grids = fhandle[os.path.join(path_inside, "grids")]
    group_densities = fhandle[os.path.join(path_inside, "densities")]

    grids = {}
    for grid in group_grids:
        m = regex_grid_name.match(grid)
        if not m:
            raise hdf5.HDF5Error("Invalid grid name \"%s\"".format(grid))
        i = int(m.group(1))
        grids[i] = group_grids[grid][:]

    densities = {}
    for density in group_densities:
        m = regex_density_name.match(density)
        if not m:
            raise hdf5.HDF5Error("Invalid density name \"%s\"".format(density))
        i = int(m.group(1))
        densities[i] = group_densities[density][:]

    fhandle.close()

    return times, grids, densities


def add_gpop_to_hdf5(group, gpop_path):
    logger = log.get_logger(__name__)

    opened_file = isinstance(group, str)
    if opened_file:
        logger.info("open new hdf5 file %s", group)
        group = h5py.File(group, "w")
    else:
        group = group.create_group("gpop")

    times, grids, densities = read_gpop(gpop_path)

    group_grids = group.create_group("grids")
    group_densities = group.create_group("densities")

    logger.info("add times")
    dset_times = group.create_dataset(
        "times", times.shape, dtype=numpy.float64, compression="gzip")
    dset_times[:] = times

    for dof in grids:
        logger.info("add grid %d", dof)
        dset_grid = group_grids.create_dataset(
            "grid" + str(dof),
            (len(grids[dof]), ),
            dtype=numpy.float64,
            compression="gzip",
        )
        dset_grid[:] = grids[dof]

        logger.info("add density %d", dof)
        dset_density = group_densities.create_dataset(
            "density" + str(dof),
            densities[dof].shape,
            dtype=numpy.float64,
            compression="gzip",
        )
        dset_density[:] = densities[dof]

    if opened_file:
        logger.info("close hdf5 file")
        group.close()
