import numpy
import pandas
import re
from mlxtk.stringio import StringIO


def read_gpop(path):
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
            dof, grid_points = fhandle.readline().split()
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
                grids[dof] = data["grid"].as_matrix()
                densities[dof] = [data["density"].as_matrix()]
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
