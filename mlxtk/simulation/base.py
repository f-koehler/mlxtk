import os
import sys
from pathlib import Path
from typing import List, Optional

from mlxtk.inout.output import read_output_ascii
from mlxtk.log import get_logger


class SimulationBase:
    def __init__(self, name: Path, working_dir: Optional[Path] = None):
        self.name = name
        self.working_dir = (
            Path(name) if working_dir is None else working_dir
        ).resolve()
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
