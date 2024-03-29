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
    x1: numpy.ndarray,
    x2: numpy.ndarray,
    values: numpy.ndarray,
    valmin: float,
    valmax: float,
    file_name_fmt: str,
    args: argparse.Namespace,
):
    unitsys = mlxtk.units.get_default_unit_system()

    X2, X1 = numpy.meshgrid(x2, x1)

    fig, ax = plt.subplots(1, 1)
    ax.set_title("$t=" + f"{time:8.2f}" + r"\," + str(unitsys.get_time_unit()) + "$")
    ax.set_xlabel(unitsys.get_length_unit().format_label("x_1"))
    ax.set_ylabel(unitsys.get_length_unit().format_label("x_2"))
    ax.set_title(
        r"$\left|g_1(x_1,x_2,t)\right|,\quad "
        + "t="
        + f"{time:8.2f}"
        + r"\,"
        + str(unitsys.get_time_unit())
        + "$",
    )
    mesh = ax.pcolormesh(
        X1,
        X2,
        values,
        rasterized=True,
        cmap="gnuplot",
        vmin=valmin,
        vmax=valmax,
    )
    fig.colorbar(mesh)
    mlxtk.plot.apply_2d_args(ax, fig, args)
    mlxtk.plot.save(fig, file_name_fmt.format(index))
    mlxtk.plot.close_figure(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        dest="input_",
        type=Path,
        default="g1.h5",
        help="path to the g1 file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("g1.mp4"),
        help="path for the video file",
    )
    mlxtk.plot.add_argparse_2d_args(parser)
    args = parser.parse_args()

    args.output = args.output.resolve()
    path_input = args.input_.resolve()

    times, x1, x2, values = mlxtk.inout.g1.read_g1_hdf5(args.input_)
    values = numpy.abs(values)

    valmin = 0.0
    valmax = values.max()

    file_name_fmt = "{:0" + str(len(str(len(times)))) + "d}.png"

    with tempfile.TemporaryDirectory() as tmpdir:
        parameters = [
            (index, time, x1, x2, density, valmin, valmax, file_name_fmt, args)
            for index, (time, density) in enumerate(zip(times, values))
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
