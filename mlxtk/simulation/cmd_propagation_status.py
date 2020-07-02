import argparse
import pickle

from mlxtk.inout.output import read_output_ascii
from mlxtk.simulation.base import SimulationBase


def check_propagation_status(self: SimulationBase, propagation: str) -> float:
    work_dir = self.working_dir / ("." + propagation)  # sim/.propagate/
    final_dir = self.working_dir / propagation  # sim/propagate/
    result_file = final_dir / "propagate.h5"  # sim/propagate/propagate.h5
    pickle_file = self.working_dir / (
        propagation + ".prop_pickle"
    )  # sim/propagate.prop_pickle
    output_file = work_dir / "output"  # sim/.propagate/output

    if not work_dir.exists():
        if result_file.exists():
            return 1.0
        else:
            return 0.0

    if not pickle_file.exists():
        return 0.0

    if not output_file.exists():
        return 0.0

    with open(pickle_file, "rb") as fptr:
        tfinal = pickle.load(fptr)[3]["tfinal"]

    times, _, _, _ = read_output_ascii(output_file)

    return times[-1] / tfinal


def cmd_propagation_status(self: SimulationBase, args: argparse.Namespace):
    self.logger.info("check progress of propagation: %s", args.name)
    progress = check_propagation_status(self, args.name)
    self.logger.info("total progress: %6.2f%%", progress * 100.0)
