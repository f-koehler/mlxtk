import argparse
from typing import Any, Callable, Dict, List

from mlxtk.doit_compat import DoitAction, run_doit
from mlxtk.simulation import Simulation
from mlxtk.simulation_set.base import SimulationSetBase


def clean_simulation(simulation: Simulation) -> List[Callable[[], Dict[str, Any]]]:
    def task_clean_simulation():
        @DoitAction
        def action_clean_simulation(targets: List[str]):
            del targets
            simulation.main(["clean"])

        task_name = simulation.name.replace("=", ":")

        return {
            "name": "clean_simulation:{}".format(task_name),
            "actions": [action_clean_simulation],
        }

    return [task_clean_simulation]


def cmd_clean(self: SimulationSetBase, args: argparse.Namespace):
    self.logger.info("cleaning simulation set")

    self.create_working_dir()

    tasks = []  # type: List[Callable[[], Dict[str, Any]]]
    for simulation in self.simulations:
        tasks += clean_simulation(simulation)
    run_doit(
        tasks,
        ["--process=" + str(args.jobs), "--backend=sqlite3", "--db-file=:memory:",],
    )
