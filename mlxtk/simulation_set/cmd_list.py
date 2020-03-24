import argparse

from mlxtk.simulation_set.base import SimulationSetBase


def cmd_list(self: SimulationSetBase, args: argparse.Namespace):
    if args.directory:
        for i, simulation in enumerate(self.simulations):
            print(i, simulation.working_dir)
    else:
        for i, simulation in enumerate(self.simulations):
            print(i, simulation.name)
