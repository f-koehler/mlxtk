import pandas

from . import hdf5


def read_dmat2(path):
    parsed = hdf5.parse_hdf5_path(path)
    if parsed is None:
        return read_dmat2_ascii(path)
    else:
        read_dmat2_hdf5(parsed)


def read_dmat2_ascii(path):
    data = pandas.read_csv(
        path,
        delimiter=r"\s+",
        header=None,
        names=["time", "dof1", "dof2", "element"])
    # time = data["time"][0]
    data.drop("time", 1, inplace=True)
    return data


def read_dmat2_hdf5(parsed_path):
    path, path_inside = parsed_path
    if not hdf5.is_hdf5_group(path, path_inside):
        raise RuntimeError("Expected a group containing the output")
