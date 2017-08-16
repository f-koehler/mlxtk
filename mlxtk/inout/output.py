import os.path

import pandas


def read_output(path):
    _, ext = os.path.splitext(path)
    if ext == ".gz":
        data = pandas.read_csv(
            path,
            compression="gzip",
            sep=r"\s+",
            names=["time", "norm", "energy", "overlap"])
    else:
        data = pandas.read_csv(
            path, sep=r"\s+", names=["time", "norm", "energy", "overlap"])

    return data
