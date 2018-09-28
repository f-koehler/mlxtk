import io
import re

import h5py
import numpy
import pandas


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
                raise RuntimeError("Failed to determine DOF and number of grid points")
            dof = int(dof)
            grid_points = int(grid_points)

            sio = io.StringIO()
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

    return [numpy.array(times), grids, densities]


def read_gpop_hdf5(path, dof=None):
    with h5py.File(path, "r") as fp:
        time = fp["time"][:]
        if dof is not None:
            dof_str = "dof_" + str(dof)
            return [time, fp[dof_str]["grid"][:], fp[dof_str]["density"][:, :]]

        grids = {}
        densities = {}
        for dof_str in (key for key in fp.keys() if key.startswith("dof_")):
            dof_i = dof_str.replace("dof_", "")
            grids[dof_i] = fp[dof_str][:]
            densities[dof_i] = fp[dof_str][:, :]
        return [time, grids, densities]


def write_gpop_hdf5(path, data):
    time, grids, densities = data
    with h5py.File(path, "w") as fp:
        dset = fp.create_dataset(
            "time", time.shape, dtype=numpy.float64, compression="gzip"
        )
        dset[:] = time

        for dof in densities:
            grp = fp.create_group("dof_" + str(dof))

            dset = grp.create_dataset(
                "grid", grids[dof].shape, dtype=numpy.float64, compression="gzip"
            )
            dset[:] = grids[dof]

            dset = grp.create_dataset(
                "density", densities[dof].shape, dtype=numpy.float64, compression="gzip"
            )
            dset[:, :] = densities[dof]
