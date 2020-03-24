import argparse

from mlxtk.simulation_set.base import SimulationSetBase


def cmd_list_tasks(self: SimulationSetBase, args: argparse.Namespace):
    self.simulations[args.index].main(["list"])
