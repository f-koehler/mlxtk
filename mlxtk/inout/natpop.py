import numpy
import os.path
import pandas
import re


def read_raw(path):
    re_time = re.compile(r"^#time:\s+(.+)\s+\[au\]$")
    re_node = re.compile(r"^node:\s+\d+\s+layer:\s+\d+$")
    re_m = re.compile(r"^m(\d+):$")

    times = []
    natpops = {}

    current_id = None
    current_chunk = None

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            splt = line.split()

            # skip empty lines
            if splt == []:
                continue

            # skip "Natural weights *1000" lines
            if splt[0] == "Natural":
                continue

            # read time stamps
            m = re_time.match(line)
            if m:
                times.append(float(m.group(1)))
                continue

            # skip "node: x layer: y" lines
            m = re_node.match(line)
            if m:
                continue

            # check if line is beginning of a new chunk
            m = re_m.match(splt[0])
            if m:
                if current_id is not None:
                    # make sure a list for the current id is in the dict
                    if current_id not in natpops:
                        natpops[current_id] = []

                    # add data of current chunk to dict
                    current_chunk = [float(x) for x in current_chunk]
                    natpops[current_id].append(current_chunk)
                    pass

                # start a new chunk
                current_id = int(m.group(1)) - 1
                current_chunk = splt[1:]
                continue

            # extend current_chunk
            current_chunk += splt

    # finish last chunk
    if current_id is not None:
        # make sure a list for the current id is in the dict
        if current_id not in natpops:
            natpops[current_id] = []

        # add data of current chunk to dict
        current_chunk = [float(x) for x in current_chunk]
        natpops[current_id].append(current_chunk)
        pass

    # start a new chunk
    current_id = int(m.group(1)) - 1
    current_chunk = splt[1:]

    # convert to data frames
    for id in natpops:
        names = ["time"] + [
            "orbital_{}".format(i) for i in range(0, len(natpops[id][0]))
        ]

        natpops[id] = numpy.array(natpops[id])
        shape = natpops[id].shape
        new_matrix = numpy.zeros(
            (shape[0], shape[1] + 1), dtype=natpops[id].dtype)
        new_matrix[:, 1:] = natpops[id] / 1000
        new_matrix[:, 0] = times

        natpops[id] = pandas.DataFrame(data=new_matrix, columns=names)
        natpops[id].transpose()

    return [natpops[id] for id in natpops]


def write(values, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i, dataset in enumerate(values):
        path = os.path.join(output_dir, "natpop_{}.gz".format(i))
        dataset.to_csv(path, index=False, compression="gzip")


def read(dir, id):
    data = pandas.read_csv(
        os.path.join(dir, "natpop_{}.gz".format(id)), compression="gzip")
    return data
