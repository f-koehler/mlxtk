from pathlib import Path
from typing import Union

import numpy

from ..inout.expval import read_expval_hdf5
from ..util import make_path
from .collect import collect_values


def collect_final_expval(scan_dir: Union[Path, str],
                         expval: Union[Path, str],
                         output_file: Union[Path, str] = None,
                         node: int = 1,
                         dof: int = 1,
                         missing_ok: bool = True):
    expval = make_path(expval)

    if output_file is None:
        folder_name = "expval_" + expval.name.rstrip(".exp.h5")
        if not folder_name.startswith("final_"):
            folder_name = "final_" + folder_name
        output_file = Path("data") / (folder_name) / (
            make_path(scan_dir).name + ".txt")

    def fetch(index, path, parameters):
        _, data = numpy.array(read_expval_hdf5(path / expval))
        return data[-1].real, data[-1].imag

    return collect_values(scan_dir, [expval],
                          output_file,
                          fetch,
                          missing_ok=missing_ok)
