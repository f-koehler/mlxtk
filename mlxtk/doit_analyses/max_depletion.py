from pathlib import Path
from typing import Union

from ..inout.natpop import read_natpop_hdf5
from ..util import make_path
from .collect import collect_values


def collect_max_depletion(scan_dir: Union[Path, str],
                          output_file: Union[Path, str] = None,
                          propagation_name: str = "propagate",
                          node: int = 1,
                          dof: int = 1,
                          missing_ok: bool = True):
    if output_file is None:
        output_file = Path("data") / ("max_depletion_{}.txt".format(
            make_path(scan_dir).name))

    def fetch(index, path, parameters):
        _, data = read_natpop_hdf5(path / propagation_name / "propagate.h5",
                                   "natpop",
                                   node=node,
                                   dof=dof)
        return (1 - data[:, 0]).max()

    return collect_values(scan_dir, [
        Path(propagation_name) / "propagate.h5",
    ],
                          output_file,
                          fetch,
                          missing_ok=missing_ok)
