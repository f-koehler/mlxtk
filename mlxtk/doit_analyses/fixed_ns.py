from pathlib import Path
from typing import Union

from mlxtk.doit_analyses.collect import collect_values
from mlxtk.inout.fixed_ns import read_fixed_ns_total_magnitude_hdf5
from mlxtk.util import make_path


def collect_final_total_magnitude(
    scan_dir: Union[Path, str],
    fixed_ns_path: Union[str, Path],
    output_file: Union[Path, str] = None,
    missing_ok: bool = True,
):

    fixed_ns_path = make_path(fixed_ns_path).with_suffix(".fixed_ns.h5")

    if output_file is None:
        output_file = (
            Path("data")
            / "final_total_magnitude_{}".format(
                fixed_ns_path.name.replace(".fixed_ns.h5", "")
            )
            / (make_path(scan_dir).name + ".txt")
        )

    def fetch(index, path, parameters):
        _, data, _, _ = read_fixed_ns_total_magnitude_hdf5(path / fixed_ns_path)
        return data[-1]

    return collect_values(
        scan_dir, [fixed_ns_path,], output_file, fetch, missing_ok=missing_ok
    )
