import argparse
import multiprocessing
import os
import subprocess
import tempfile
from functools import partial
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy

from mlxtk import plot, units
from mlxtk.cwd import WorkingDir
from mlxtk.inout.dmat2 import read_dmat2_gridrep_hdf5
from mlxtk.log import get_logger
from mlxtk.util import copy_file

LOGGER = get_logger(__name__)


def render_frame(i, times, dmat2, X1, X2, num_digits, dpi):
    unit_system = units.get_default_unit_system()
    fig, ax = plt.subplots()
    ax.set_title("$t=" + str(times[i]) + "$")
    ax.set_xlabel(unit_system.get_length_unit().format_label("x_1"))
    ax.set_ylabel(unit_system.get_length_unit().format_label("x_2"))
    ax.pcolormesh(X1, X2, dmat2[i])
    fig.savefig(("{:0" + str(num_digits) + "d}.png").format(i), dpi=dpi)
    plot.close_figure(fig)
    LOGGER.info("rendered image %d", i)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="path for the input file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="path for the video file"
    )
    parser.add_argument(
        "-f", "--fps", type=float, default=5, help="frames per second for the video"
    )
    parser.add_argument(
        "--dpi", type=int, default=600, help="resolution (dpi) of the individual frames"
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=os.cpu_count(),
        help="number of threads to use to create the individual frames",
    )
    args = parser.parse_args()

    times, x1, x2, dmat2 = read_dmat2_gridrep_hdf5(args.input)
    X2, X1 = numpy.meshgrid(x2, x1)

    num_digits = len(str(len(times)))

    if args.output:
        output = args.output.resolve()
    else:
        output = args.input.with_suffix(".mp4").resolve()

    with tempfile.TemporaryDirectory() as tmpdir:
        with WorkingDir(tmpdir):
            with multiprocessing.Pool(args.jobs) as pool:
                pool.map(
                    partial(
                        render_frame,
                        times=times,
                        dmat2=dmat2,
                        X1=X1,
                        X2=X2,
                        num_digits=num_digits,
                        dpi=args.dpi,
                    ),
                    [i for i, _ in enumerate(times)],
                )

            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(args.fps),
                "-i",
                "%0" + str(num_digits) + "d.png",
                "out.mp4",
            ]
            LOGGER.info("ffmpeg command: %s", " ".join(cmd))
            subprocess.run(cmd)
            LOGGER.info("copy file: out.mp4 -> %s", str(output))
            copy_file("out.mp4", output)


if __name__ == "__main__":
    main()
