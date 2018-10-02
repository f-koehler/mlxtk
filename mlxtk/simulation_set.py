import argparse


class SimulationSet(object):
    def __init__(self):
        self.simulations = []

    def run(self):
        pass

    def main(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="subcommand")

        subparsers.add_parser("run")
