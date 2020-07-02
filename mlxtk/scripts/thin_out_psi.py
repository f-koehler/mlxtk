import argparse
import os.path
import re

import numpy

from mlxtk.inout import read_psi_ascii, write_psi_ascii
from mlxtk.log import get_logger

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(metavar="input", dest="input_", type=str, help="input psi file")
    parser.add_argument("-o", "--output", type=str, help="output psi file")
    parser.add_argument(
        "-t", "--time", type=float, action="append", help="time point for psi"
    )
    parser.add_argument(
        "-s",
        "--slice",
        dest="slice_",
        type=str,
        action="append",
        help="index slices for new psi",
    )
    args = parser.parse_args()

    initial_size = os.path.getsize(os.path.realpath(args.input_))
    tape, times, psi = read_psi_ascii(args.input_)
    LOGGER.info("starting with %d time steps", len(times))

    indices = set()

    if args.time:
        for time in args.time:
            LOGGER.info("add time %s", time)
            indices.add(numpy.abs(times - time).argmin())

    # TODO: I reimplemented this code as a function for the simulation_set.
    # Move that function to a separate module and reuse it here
    if args.slice_:
        re_slice = re.compile(r"^([+-]*\d*):([+-]*\d*)(?::([+-]*\d*))?$")
        for slice_ in args.slice_:
            LOGGER.info("add slice: %s", slice_)
            m = re_slice.match(slice_)
            if not m:
                raise RuntimeError('Invalid slice: "%s"', slice_)
            start = 0
            stop = len(times)
            step = 1
            if m.group(1) != "":
                start = int(m.group(1))
            if m.group(2) != "":
                stop = int(m.group(2))
            try:
                if m.group(3) != "":
                    step = int(m.group(3))
            except IndexError:
                pass

            LOGGER.info("start: %d, stop: %d, step: %d", start, stop, step)

            for i in range(start, stop, step):
                indices.add(i)

    if not indices:
        raise RuntimeError("No frames for new psi file")

    indices = list(indices)
    indices.sort()
    times = times[indices]
    psi = psi[indices]
    write_psi_ascii(args.output, (tape, times, psi))

    final_size = os.path.getsize(os.path.realpath(args.output))
    LOGGER.info("ending up with %d time steps", len(times))


if __name__ == "__main__":
    main()
