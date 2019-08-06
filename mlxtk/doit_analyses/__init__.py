from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt

from .. import inout, plot
from ..doit_compat import DoitAction
from ..parameter_selection import load_scan
from ..util import make_path


def scan_plot_gpop(scan_dirs: Union[Path, str, List[str], List[Path]],
                   propagation: str = "propagate",
                   dof: int = 1,
                   extensions: List[str] = [".png", ".pdf"],
                   output_dir: str = None):
    if isinstance(scan_dirs, list):
        scan_dirs = [make_path(p) for p in scan_dirs]
    else:
        scan_dirs = [make_path(scan_dirs)]

    for scan_dir in scan_dirs:
        selection = load_scan(scan_dir)
        for (index, parameters), path in zip(selection.parameters,
                                             selection.get_paths()):
            data_file = scan_dir / "by_index" / str(
                index) / propagation / "propagate.h5"

            output_files = []
            if output_dir:
                for extension in extensions:
                    output_files.append(scan_dir / "plots" / output_dir /
                                        (str(index) + extension))
            else:
                for extension in extensions:
                    output_files.append(scan_dir / "plots" /
                                        "gpop_{}".format(dof) /
                                        (str(index) + extension))

            @DoitAction
            def action_plot(targets):
                output_files = [Path(p) for p in targets]
                if not output_files[0].parent.exists():
                    output_files[0].parent.mkdir(parents=True)

                time, grid, density = inout.gpop.read_gpop(str(data_file) +
                                                           "/gpop",
                                                           dof=dof)

                fig, ax = plt.subplots(1, 1)
                plot.gpop.plot_gpop(ax, time, grid, density)
                for output_file in output_files:
                    plot.save(fig, output_file)

            yield {
                "name":
                "{}:{}:gpop:index_{}:dof_{}".format(str(scan_dir), propagation,
                                                    index, dof),
                "actions": [action_plot],
                "targets": [str(p) for p in output_files],
                "file_dep": [str(data_file)],
                "clean":
                True
            }
