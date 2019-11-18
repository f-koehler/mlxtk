from pathlib import Path
from typing import Tuple, Union

import h5py
import numpy

from ..util import make_path


def add_g1_to_hdf5(
        fptr: Union[h5py.File, h5py.Group],
        data: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
):
    group = fptr.create_group("g1")
    group.create_dataset("time", data[0].shape, data[0].dtype)[:] = data[0]
    group.create_dataset("x1", data[1].shape, data[1].dtype)[:] = data[1]
    group.create_dataset("x2", data[2].shape, data[2].dtype)[:] = data[2]
    group.create_dataset("real", data[3].shape,
                         data[3].real.dtype)[:] = data[3].real
    group.create_dataset("imag", data[3].shape,
                         data[3].imag.dtype)[:] = data[3].imag
