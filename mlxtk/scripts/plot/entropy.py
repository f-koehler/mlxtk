import argparse

import matplotlib.pyplot as plt

from mlxtk import units
from mlxtk.inout import read_natpop
from mlxtk.plot import (
    add_argparse_2d_args,
    add_argparse_save_arg,
    apply_2d_args,
    handle_saving,
    plot_entropy,
)
from mlxtk.tools.entropy import compute_entropy
from mlxtk.util import labels_from_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", nargs="*", default=["natpop"], help="path to the natpop file"
    )
    parser.add_argument("-n", "--node", type=int, default=1, help="node")
    parser.add_argument("-d", "--dof", type=int, default=1, help="degree of freedom")
    add_argparse_2d_args(parser)
    add_argparse_save_arg(parser)
    args = parser.parse_args()

    labels = labels_from_paths(args.path)
    figure, ax = plt.subplots(1, 1)

    for path, label in zip(args.path, labels):
        time, natpop = read_natpop(path, node=args.node, dof=args.dof)
        entropy = compute_entropy(natpop)
        plot_entropy(ax, time, entropy, label=label)

    system = units.get_default_unit_system()
    ax.set_xlabel(system.get_time_unit().format_label("t"))

    apply_2d_args(ax, figure, args)
    if len(args.path) > 1:
        ax.legend()

    handle_saving(figure, args)
    plt.show()


if __name__ == "__main__":
    main()
