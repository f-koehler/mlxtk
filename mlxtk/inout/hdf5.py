import os.path
import h5py


class HDF5Error(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)


class IncompleteHDF5(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)


def parse_hdf5_path(path):
    path_inside = None

    while path not in ["/", ""]:
        if is_hdf5_file(path):
            return path, os.path.join("/", path_inside)

        path, tail = os.path.split(path)
        if path_inside is None:
            path_inside = tail
        else:
            path_inside = os.path.join(tail, path_inside)

    return None


def is_hdf5_dataset(path, path_inside):
    fhandle = h5py.File(path, "r")
    result = isinstance(fhandle[path_inside], h5py.Dataset)
    fhandle.close()
    return result


def is_hdf5_group(path, path_inside):
    fhandle = h5py.File(path, "r")
    result = isinstance(fhandle[path_inside], h5py.Group)
    fhandle.close()
    return result


def is_hdf5_file(path):
    magic_number = bytes([0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a])
    if not os.path.exists(path):
        return False

    if not os.path.isfile(path):
        return False

    with open(path, "rb") as fhandle:
        result = fhandle.read(8) == magic_number

    return result
