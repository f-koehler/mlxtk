import argparse


def cmd_list(self, args: argparse.Namespace):
    if args.directory:
        for i, simulation in enumerate(self.simulations):
            print(i, simulation.working_dir)
    else:
        for i, simulation in enumerate(self.simulations):
            print(i, simulation.name)
