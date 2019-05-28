import argparse
import os.path

import numpy

from ..inout import read_psi_ascii, write_psi_ascii
from ..log import get_logger

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(metavar="input",
                        dest="input_",
                        type=str,
                        help="input psi file")
    parser.add_argument("output", type=str, help="output psi file")
    parser.add_argument("--imin",
                        default=0,
                        type=int,
                        help="starting index for new psi")
    parser.add_argument("--imax", type=int, help="stopping index for new psi")
    parser.add_argument("--di",
                        default=2,
                        type=int,
                        help="index stride for new psi")
    args = parser.parse_args()

    tape, times, psi = read_psi_ascii(args.input_)
    initial_size = os.path.getsize(args.input_)
    LOGGER.info("starting with %d time steps", len(times))

    if args.imax is None:
        args.imax = len(times)

    times = times[args.imin:args.imax:args.di]
    psi = psi[args.imin:args.imax:args.di]
    LOGGER.info("ending up with %d time steps", len(times))
    write_psi_ascii(args.output, (tape, times, psi))
    final_size = os.path.getsize(args.output)
    LOGGER.info("file size reduced by {:.2f}%".format(
        (1 - final_size / initial_size) * 100.))


if __name__ == "__main__":
    main()
