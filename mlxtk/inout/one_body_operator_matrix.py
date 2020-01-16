from pathlib import Path
from typing import Tuple, Union

import h5py
import numpy

from mlxtk.util import make_path


def read_one_body_operator_matrix(
    path: Union[Path, str]
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    with h5py.File(make_path(path), "r") as fptr:
        return fptr["grid_1"][:], fptr["weights_1"][:], fptr["matrix"][:, :]
