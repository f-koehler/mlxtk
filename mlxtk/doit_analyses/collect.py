from pathlib import Path
from typing import List, Union

import numpy

from mlxtk.log import get_logger
from mlxtk.parameter_selection import load_scan
from mlxtk.util import make_path

LOGGER = get_logger(__name__)


def collect_values(
    scan_dir: Union[Path, str],
    data_files: List[Union[Path, str]],
    output_file: Union[Path, str],
    fetch_func,
    missing_ok: bool = True,
):
    scan_dir = make_path(scan_dir)
    data_files = [make_path(p) for p in data_files]
    output_file = make_path(output_file)

    selection = load_scan(scan_dir)
    file_deps = []
    for data_file in data_files:
        for i, _ in selection.parameters:
            p = scan_dir / "by_index" / str(i) / data_file
            if missing_ok and p.exists():
                file_deps.append(p)

    def action_collect_values(scan_dir: Path, targets):
        selection = load_scan(scan_dir)
        variables = selection.get_variable_names()

        def helper(index, path, parameters):
            return (
                [parameters[variable] for variable in variables],
                fetch_func(index, path, parameters),
            )

        parameters = []
        values_lst = []
        for param, val in selection.foreach(helper, parallel=False):
            if val is not None:
                parameters.append(param)
                values_lst.append(val)
            else:
                LOGGER.warning("cannot fetch value(s) for parameters: %s", str(param))

        parameters = numpy.array(parameters, dtype=object)
        values = numpy.array(values_lst, dtype=object)
        if len(values.shape) == 1:
            values = values.reshape((len(values), 1))
        elif len(values.shape) == 2:
            pass
        else:
            raise RuntimeError("Invalid dimensions {}".format(len(values.shape)))

        data = numpy.c_[parameters, values]
        header = [variable for variable in variables] + [
            "value{}".format(i) for i in range(values.shape[1])
        ]
        Path(targets[0]).parent.mkdir(parents=True, exist_ok=True)
        numpy.savetxt(targets[0], data, header=" ".join(header))

    yield {
        "name": "{}:collect_values:{}".format(
            str(scan_dir.name), str(output_file.with_suffix(""))
        )
        .replace("=", "_")
        .replace("/", "_"),
        "targets": [str(output_file)],
        "file_dep": file_deps,
        "clean": True,
        "actions": [(action_collect_values, [scan_dir])],
    }
