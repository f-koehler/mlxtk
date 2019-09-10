import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy
from tqdm import tqdm

from .. import plot, units
from ..cwd import WorkingDir
from ..inout.dmat2 import read_dmat2_gridrep_hdf5
from ..util import copy_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="path for the input file")
    parser.add_argument("-o",
                        "--output",
                        type=Path,
                        default=None,
                        help="path for the video file")
    parser.add_argument("-f",
                        "--fps",
                        type=float,
                        default=5,
                        help="frames per second for the video")
    args = parser.parse_args()

    unit_system = units.get_default_unit_system()
    times, x1, x2, dmat2 = read_dmat2_gridrep_hdf5(args.input)
    X2, X1 = numpy.meshgrid(x2, x1)

    num_digits = len(str(len(times)))

    if args.output:
        output = args.output.resolve()
    else:
        output = args.input.with_suffix(".mp4")

    with tempfile.TemporaryDirectory() as tmpdir:
        with WorkingDir(tmpdir):
            for i, (time, mat) in tqdm(enumerate(zip(times, dmat2)),
                                       total=len(times)):
                fig, ax = plt.subplots()
                ax.set_title("$t=" + str(time) + "$")
                ax.set_xlabel(
                    unit_system.get_length_unit().format_label("x_1"))
                ax.set_ylabel(
                    unit_system.get_length_unit().format_label("x_2"))
                ax.pcolormesh(X1, X2, mat)
                fig.savefig(("{:0" + str(num_digits) + "d}.png").format(i),
                            dpi=600)
                plot.close_figure(fig)

            subprocess.run([
                "ffmpeg", "-y", "-framerate",
                str(args.fps), "-i", "%0" + str(num_digits) + "d.png",
                "out.mp4"
            ])
            copy_file("out.mp4", output)


if __name__ == "__main__":
    main()
