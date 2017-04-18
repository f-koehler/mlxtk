import os.path
import pandas


def read_raw(input_file):
    t = []

    new_block = False
    values = []

    with open(input_file) as fh:
        while True:
            line = fh.readline()
            if line == '':
                # an empty string can only occur at EOF
                break

            spl = line.split()

            if not new_block:
                if spl == []:
                    # found a blank line, a new block will start
                    new_block = True
                    continue
                if spl[0] == "#":
                    # found a time stamp, a new block will start
                    # extract time
                    t.append(float(spl[1]))
                    new_block = True
                    continue
                raise ValueError("Expected either a blank line or a time stamp")

            # read a new block of data
            dof = int(spl[0]) - 1
            grid_size = int(spl[1])

            x = []
            density = [t[-1]]
            for i in range(0, grid_size):
                spl = fh.readline().split()
                x.append(float(spl[0]))
                density.append(float(spl[1]))

            if dof >= len(values):
                # new dof
                values.append({
                    "dof": dof,
                    "grid_size": grid_size,
                    "grid": x,
                    "densities": [density]
                })
            else:
                # know dof
                values[dof]["densities"].append(density)

            new_block = False
            fh.readline()

    for v in values:
        names = ["time"] + [
            "x_{}".format(i) for i in range(0, len(v["densities"][1]) - 1)
        ]
        v["grid"] = pandas.DataFrame(v["grid"], columns=["x"])
        v["densities"] = pandas.DataFrame(v["densities"], columns=names)

    return values


def write(values, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for dataset in values:
        grid_file = os.path.join(output_dir,
                                 "grid_{}.gz".format(dataset["dof"]))
        density_file = os.path.join(output_dir,
                                    "density_{}.gz".format(dataset["dof"]))

        dataset["grid"].to_csv(grid_file, index=False, compression="gzip")
        dataset["densities"].to_csv(
            density_file, index=False, compression="gzip")


def read(dir, dof):
    grid_file = os.path.join(dir, "grid_{}.gz".format(dof))
    density_file = os.path.join(dir, "density_{}.gz".format(dof))

    grid = pandas.read_csv(grid_file, compression="gzip")
    density = pandas.read_csv(density_file, compression="gzip")

    return grid, density
