"""Work with a collection of simulations.
"""
import argparse
import os
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional

from . import cwd, doit_compat, sge
from .log import get_logger
from .simulation import Simulation

assert Any
assert Callable
assert Dict

LOGGER = get_logger(__name__)


def run_simulation(simulation: Simulation) -> List[Dict[str, Any]]:
    """Create a task for running the given simulation.

    Args:
        simulation (Simulation): simulation to run

    Returns:
        list: list with a single doit task description for simulation run
    """

    def task_run_simulation():
        def action_run_simulation(targets):
            del targets
            simulation.main(["run"])

        task_name = simulation.name.replace("=", ":")

        return {
            "name": "run_simulation:{}".format(task_name),
            "actions": [action_run_simulation],
        }

    return [task_run_simulation]


class SimulationSet:
    """A collection of simulations.

    Args:
        name (str): name for this set of simulations
        simulations (list): the list of simulations
        working_dir (str): optional working directory

    Attributes:
        name (str): name for this set of simulations
        simulations (list): the list of simulations
        working_dir (str): optional working directory (defaults to ``name``)
    """

    def __init__(
            self,
            name: str,
            simulations: List[Simulation],
            working_dir: Optional[str]=None, ):
        self.name = name
        self.simulations = simulations
        self.working_dir = name if working_dir is None else working_dir

        self.argparser = argparse.ArgumentParser(
            description="This is a set of mlxtk simulations")
        subparsers = self.argparser.add_subparsers(dest="subcommand")

        subparsers.add_parser("list")
        self.argparser_list_tasks = subparsers.add_parser("list-tasks")
        self.argparser_task_info = subparsers.add_parser("task-info")
        subparsers.add_parser("qdel")
        self.argparser_qsub = subparsers.add_parser("qsub")
        self.argparser_run = subparsers.add_parser("run")
        self.argparser_run_index = subparsers.add_parser("run-index")

        sge.add_parser_arguments(self.argparser_qsub)

        self.argparser_list_tasks.add_argument(
            "index",
            type=int,
            help="index of the simulation whose tasks to list")
        self.argparser_task_info.add_argument(
            "index", type=int, help="index of the simulation")
        self.argparser_task_info.add_argument(
            "name", type=str, help="name of the task")
        self.argparser_run.add_argument(
            "-j",
            "--jobs",
            type=int,
            default=1,
            help="number of parallel workers")
        self.argparser_run_index.add_argument(
            "index", type=int, help="index of the simulation to run")

    def create_working_dir(self):
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

    def list_(self, args: argparse.Namespace):
        del args

        for i, simulation in enumerate(self.simulations):
            print(i, simulation.name)

    def list_tasks(self, args: argparse.Namespace):
        self.simulations[args.index].main(["list"])

    def task_info(self, args: argparse.Namespace):
        self.create_working_dir()

        with cwd.WorkingDir(self.working_dir):
            self.simulations[args.index].main(["task-info", args.name])

    def run(self, args: argparse.Namespace):
        self.create_working_dir()

        with cwd.WorkingDir(self.working_dir):
            tasks = []  # type: List[Callable[[], Dict[str, Any]]]
            for simulation in self.simulations:
                tasks += run_simulation(simulation)
            doit_compat.run_doit(
                tasks,
                [
                    "-n",
                    str(args.jobs),
                    "--backend",
                    "sqlite3",
                    "--db-file",
                    "doit.sqlite3",
                ], )

    def run_index(self, args: argparse.Namespace):
        self.create_working_dir()

        with cwd.WorkingDir(self.working_dir):
            self.simulations[args.index].main(["run"])

    def qdel(self, args: argparse.Namespace):
        del args

        if not os.path.exists(self.working_dir):
            LOGGER.warning("working dir %s does not exist, do nothing",
                           self.working_dir)

        with cwd.WorkingDir(self.working_dir):
            for simulation in self.simulations:
                simulation.main(["qdel"])

    def qsub(self, args: argparse.Namespace):
        self.create_working_dir()

        set_dir = os.path.abspath(self.working_dir)
        script_dir = os.path.abspath(sys.argv[0])

        for index, simulation in enumerate(self.simulations):
            simulation_dir = os.path.join(set_dir, simulation.working_dir)
            if not os.path.exists(simulation_dir):
                os.makedirs(simulation_dir)
            with cwd.WorkingDir(simulation_dir):
                sge.submit(
                    " ".join(
                        [sys.executable, script_dir, "run-index",
                         str(index)]),
                    args,
                    sge_dir=simulation_dir,
                    job_name=simulation.name)

    def main(self, args: Iterable[str]=sys.argv[1:]):

        args = self.argparser.parse_args(args)

        if args.subcommand is None:
            LOGGER.error("No subcommand specified!")
            print()
            self.argparser.print_help(sys.stderr)
            exit(1)

        if args.subcommand == "list":
            self.list_(args)
        if args.subcommand == "list-tasks":
            self.list_tasks(args)
        elif args.subcommand == "qdel":
            self.qdel(args)
        elif args.subcommand == "qsub":
            self.qsub(args)
        elif args.subcommand == "run":
            self.run(args)
        elif args.subcommand == "task-info":
            self.task_info(args)
