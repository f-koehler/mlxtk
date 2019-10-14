import argparse


def cmd_list_tasks(self, args: argparse.Namespace):
    self.simulations[args.index].main(["list"])
