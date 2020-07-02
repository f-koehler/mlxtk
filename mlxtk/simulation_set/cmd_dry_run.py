import argparse
from typing import Any, Callable, Dict, List

from mlxtk.doit_compat import DoitAction, run_doit
from mlxtk.simulation import Simulation
from mlxtk.simulation_set.base import SimulationSetBase


def dry_run_simulation(simulation: Simulation) -> List[Callable[[], Dict[str, Any]]]:
    """Create a task for running the given simulation dry.

    Args:
        simulation (Simulation): simulation to run

    Returns:
        list: list with a single doit task description for simulation run
    """

    def task_dry_run_simulation():
        @DoitAction
        def action_dry_run_simulation(targets: List[str]):
            del targets
            simulation.main(["dry-run"])

        task_name = simulation.name.replace("=", ":")

        return {
            "name": "dry_run_simulation:{}".format(task_name),
            "actions": [action_dry_run_simulation],
        }

    return [task_dry_run_simulation]


def cmd_dry_run(self: SimulationSetBase, args: argparse.Namespace):
    del args

    self.logger.info("running simulation set (dry)")

    self.create_working_dir()

    tasks = []  # type: List[Callable[[], Dict[str, Any]]]
    for simulation in self.simulations:
        tasks += dry_run_simulation(simulation)
    run_doit(
        tasks, ["--backend=sqlite3", "--db-file=:memory:",],
    )
