import argparse

import h5py
import numpy
import tabulate

from ..tools.wave_function import build_number_state_table_bosonic
from ..util import compute_magnitude_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", help="result of the ns analysis")
    parser.add_argument("--sort",
                        action="store_true",
                        default=False,
                        help="whether to sort by magnitude")
    args = parser.parse_args()

    with h5py.File(args.result, "r") as fp:
        times = fp["fixed_ns"]["time"][:]
        if len(times) != 1:
            raise RuntimeError(
                "this script expects exactly one time instance in the ns analysis"
            )

        magnitudes = compute_magnitude_split(
            fp["fixed_ns"]["real"][:, :].flatten(),
            fp["fixed_ns"]["imag"][:, :].flatten())
        N = fp["fixed_ns"].attrs["N"]
        m = fp["fixed_ns"].attrs["m"]

    states = [
        "|" + " ".join([str(n) for n in ns]) + "⟩"
        for ns in build_number_state_table_bosonic(N, m)
    ]

    if args.sort:
        indices = numpy.argsort(magnitudes)[::-1]
        magnitudes = magnitudes[indices]
        states = numpy.array(states)[indices].tolist()

    print("Sum:", numpy.sum(magnitudes))
    print()

    print(
        tabulate.tabulate({
            "states": states,
            "magnitudes": magnitudes
        },
                          headers="keys",
                          tablefmt="orgtbl"))


if __name__ == "__main__":
    main()
