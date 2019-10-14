import os
import pickle
from pathlib import Path
from typing import Optional

from ..inout.output import read_output_ascii
from ..log import get_logger


class SimulationBase:
    def __init__(self, name: Path, working_dir: Optional[Path] = None):
        self.name = name
        self.working_dir = (Path(name)
                            if working_dir is None else working_dir).resolve()
        self.tasks_run = []
        self.tasks_clean = []
        self.tasks_dry_run = []
        self.logger = get_logger(__name__ + ".Simulation")

    def __iadd__(self, generator):
        self.tasks_run += generator()
        self.tasks_clean += generator.get_tasks_clean()
        self.tasks_dry_run += generator.get_tasks_dry_run()
        return self

    def create_working_dir(self):
        if not self.working_dir.exists():
            os.makedirs(self.working_dir)

    def check_propagation_status(self, propagation: str) -> float:
        work_dir = self.working_dir / ("." + propagation)  # sim/.propagate/
        final_dir = self.working_dir / propagation  # sim/propagate/
        result_file = final_dir / "propagate.h5"  # sim/propagate/propagate.h5
        pickle_file = self.working_dir / (propagation + ".prop_pickle"
                                          )  # sim/propagate.prop_pickle
        output_file = work_dir / "output"  # sim/.propagate/output

        if not work_dir.exists():
            if result_file.exists():
                return 1.
            else:
                return 0.

        if not pickle_file.exists():
            return 0.

        if not output_file.exists():
            return 0.

        with open(pickle_file, "rb") as fptr:
            tfinal = pickle.load(fptr)[3]["tfinal"]

        times, _, _, _ = read_output_ascii(output_file)

        return times[-1] / tfinal
