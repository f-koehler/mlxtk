import pandas
import os.path


def read(path):
    _, ext = os.path.splitext(path)
    if ext == ".gz":
        data = pandas.read_csv(
            path,
            compression="gzip",
            sep=r"\s+",
            names=["time", "real_part", "imaginary_part"])
    else:
        data = pandas.read_csv(
            path, sep=r"\s+", names=["time", "real", "imaginary"])

    return data
