from pathlib import Path
from typing import Union

from ..inout.output import read_output_hdf5
from ..util import make_path
from .collect import collect_values


def collect_final_energy(scan_dir: Union[Path, str],
                         propagation_name: str = "propagate",
                         output_file: Union[Path, str] = None,
                         missing_ok: bool = True):
    if output_file is None:
        output_file = Path("data") / "final_energy" / (
            make_path(scan_dir).name + ".txt")

    def fetch(index, path, parameters):
        _, _, energy, _ = read_output_hdf5(
            path / propagation_name / "propagate.h5", "output")
        return energy[-1]

    return collect_values(scan_dir, [
        Path(propagation_name) / "propagate.h5",
    ],
                          output_file,
                          fetch,
                          missing_ok=missing_ok)
