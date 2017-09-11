import os.path

import pandas


def read_output(path):
    """Read a QDTK ouput file

    The resulting :py:class:`pandas.DataFrame` contains the columns `time`,
    `norm`, `energy` and `overlap`.

    Args:
        path (str): Path to the output file

    Returns:
        pandas.DataFrame: The output of the QDTK run
    """
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
