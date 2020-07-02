import argparse
from typing import Any, Callable, Dict, List

from mlxtk.doit_compat import DoitAction, run_doit
from mlxtk.simulation import Simulation
from mlxtk.simulation_set.base import SimulationSetBase


def run_simulation(simulation: Simulation) -> List[Callable[[], Dict[str, Any]]]:
    """Create a task for running the given simulation.

    Args:
        simulation (Simulation): simulation to run

    Returns:
        list: list with a single doit task description for simulation run
    """

    def task_run_simulation():
        @DoitAction
        def action_run_simulation(targets: List[str]):
            del targets
            simulation.main(["run"])

        task_name = simulation.name.replace("=", ":")

        return {
            "name": "run_simulation:{}".format(task_name),
            "actions": [action_run_simulation],
        }

    return [task_run_simulation]


def cmd_run(self: SimulationSetBase, args: argparse.Namespace):
    self.logger.info("running simulation set")

    self.create_working_dir()

    tasks = []  # type: List[Callable[[], Dict[str, Any]]]
    for simulation in self.simulations:
        tasks += run_simulation(simulation)
    run_doit(
        tasks,
        ["--process=" + str(args.jobs), "--backend=sqlite3", "--db-file=:memory:",],
    )
