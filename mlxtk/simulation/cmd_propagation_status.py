import argparse


def cmd_propagation_status(self, args: argparse.Namespace):
    self.logger.info("check progress of propagation: %s", args.name)
    progress = self.check_propagation_status(args.name)
    self.logger.info("total progress: %6.2f%%", progress * 100.)
