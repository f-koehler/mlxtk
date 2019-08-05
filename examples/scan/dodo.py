import mlxtk
from pathlib import Path
import matplotlib.pyplot as plt

scans = ["harmonic_trap_scan"]


def task_plot_gpop():
    tasks = []

    for scan in scans:
        selection = mlxtk.load_scan(scan)
        for (index, parameters), path in zip(selection.parameters,
                                             selection.get_paths()):

            data_file = Path(scan) / "by_index" / str(
                index) / "propagate" / "propagate.h5"
            output_png = Path(scan) / "plots" / "gpop_1" / (str(index) +
                                                            ".png")
            output_pdf = output_png.with_suffix(".pdf")

            @mlxtk.doit_compat.DoitAction
            def action_plot(targets):
                if not output_png.parent.exists():
                    output_png.parent.mkdir(parents=True)

                time, grid, density = mlxtk.inout.gpop.read_gpop(
                    str(data_file) + "/gpop", dof=1)

                fig, ax = plt.subplots(1, 1)
                mlxtk.plot.gpop.plot_gpop(ax, time, grid, density)
                mlxtk.plot.save(fig, targets[0])
                mlxtk.plot.save(fig, targets[1])

            yield {
                "name": "{}:plot_gpop:{}".format(scan, index),
                "actions": [action_plot],
                "targets": [output_png, output_pdf],
                "file_dep": [data_file],
                "clean": True
            }
