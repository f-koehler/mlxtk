import argparse
import multiprocessing
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
from tqdm import tqdm

import mlxtk


def create_frame_worker(args):
    create_frame(*args)


def create_frame(
    index: int,
    time: float,
    grid: numpy.ndarray,
    density: numpy.ndarray,
    file_name_fmt: str,
    args: argparse.Namespace,
):
    unitsys = mlxtk.units.get_default_unit_system()

    fig, ax = plt.subplots(1, 1)
    mlxtk.plot.apply_2d_args(ax, fig, args)
    ax.set_title(
        "$t=" + "{:8.2f}".format(time) + r"\," + str(unitsys.get_time_unit()) + "$"
    )
    ax.set_xlabel(unitsys.get_length_unit().format_label("x"))
    ax.set_ylabel(r"$\rho_1(x,t)$")
    ax.grid(True)
    ax.plot(grid, density)
    mlxtk.plot.save(fig, file_name_fmt.format(index))
    mlxtk.plot.close_figure(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        dest="input_",
        type=Path,
        default="propagate.h5/gpop",
        help="path to the gpop file",
    )
    parser.add_argument(
        "-d", "--dof", type=int, default=1, help="degree of freedom to use"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="path for the video file"
    )
    mlxtk.plot.add_argparse_2d_args(parser)
    args = parser.parse_args()

    if not args.output:
        args.output = Path("gpop_{}.mp4".format(args.dof))
    args.output = args.output.resolve()

    path_input = args.input_.resolve()

    times, grid, gpop = mlxtk.inout.gpop.read_gpop(args.input_, args.dof)

    file_name_fmt = "{:0" + str(len(str(len(times)))) + "d}.png"

    with tempfile.TemporaryDirectory() as tmpdir:
        parameters = [
            (index, time, grid, density, file_name_fmt, args)
            for index, (time, density) in enumerate(zip(times, gpop))
        ]
        with mlxtk.cwd.WorkingDir(tmpdir):
            with multiprocessing.Pool() as pool:
                for _ in tqdm(
                    pool.imap_unordered(create_frame_worker, parameters),
                    total=len(times),
                ):
                    pass

        files = [Path(tmpdir) / file_name_fmt.format(i) for i, _ in enumerate(times)]
        mlxtk.tools.video.create_slideshow(files, args.output, 30.0)


if __name__ == "__main__":
    main()
