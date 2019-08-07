"""Work with a collection of simulations.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

from . import cwd, doit_compat, sge, util
from .log import get_logger
from .simulation import Simulation
from .util import make_path

assert Any
assert Callable
assert Dict

LOGGER = get_logger(__name__)


def run_simulation(simulation: Simulation
                   ) -> List[Callable[[], Dict[str, Any]]]:
    """Create a task for running the given simulation.

    Args:
        simulation (Simulation): simulation to run

    Returns:
        list: list with a single doit task description for simulation run
    """
    def task_run_simulation():
        @doit_compat.DoitAction
        def action_run_simulation(targets: List[str]):
            del targets
            simulation.main(["run"])

        task_name = simulation.name.replace("=", ":")

        return {
            "name": "run_simulation:{}".format(task_name),
            "actions": [action_run_simulation],
        }

    return [task_run_simulation]


def dry_run_simulation(simulation: Simulation
                       ) -> List[Callable[[], Dict[str, Any]]]:
    """Create a task for running the given simulation dry.

    Args:
        simulation (Simulation): simulation to run

    Returns:
        list: list with a single doit task description for simulation run
    """
    def task_dry_run_simulation():
        @doit_compat.DoitAction
        def action_dry_run_simulation(targets: List[str]):
            del targets
            simulation.main(["dry-run"])

        task_name = simulation.name.replace("=", ":")

        return {
            "name": "dry_run_simulation:{}".format(task_name),
            "actions": [action_dry_run_simulation],
        }

    return [task_dry_run_simulation]


def clean_simulation(simulation: Simulation
                     ) -> List[Callable[[], Dict[str, Any]]]:
    def task_clean_simulation():
        @doit_compat.DoitAction
        def action_clean_simulation(targets: List[str]):
            del targets
            simulation.main(["clean"])

        task_name = simulation.name.replace("=", ":")

        return {
            "name": "clean_simulation:{}".format(task_name),
            "actions": [action_clean_simulation],
        }

    return [task_clean_simulation]


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
            working_dir: Union[str, Path] = None,
    ):
        self.name = name
        self.simulations = simulations
        self.working_dir = Path(
            name).resolve() if working_dir is None else make_path(working_dir)

        self.logger = get_logger(__name__ + ".SimulationSet")

        self.argparser = argparse.ArgumentParser(
            description="This is a set of mlxtk simulations")
        subparsers = self.argparser.add_subparsers(dest="subcommand")

        self.argparser_list = subparsers.add_parser("list")
        self.argparser_list_tasks = subparsers.add_parser("list-tasks")
        self.argparser_task_info = subparsers.add_parser("task-info")
        subparsers.add_parser("qdel")
        self.argparser_qsub = subparsers.add_parser("qsub")
        self.argparser_run = subparsers.add_parser("run")
        self.argparser_dry_run = subparsers.add_parser("dry-run")
        self.argparser_run_index = subparsers.add_parser("run-index")
        self.argparser_clean = subparsers.add_parser("clean")
        self.argparser_archive = subparsers.add_parser("archive")

        sge.add_parser_arguments(self.argparser_qsub)

        self.argparser_list.add_argument("-d",
                                         "--directory",
                                         action="store_true")

        self.argparser_list_tasks.add_argument(
            "index",
            type=int,
            help="index of the simulation whose tasks to list")
        self.argparser_task_info.add_argument("index",
                                              type=int,
                                              help="index of the simulation")
        self.argparser_task_info.add_argument("name",
                                              type=str,
                                              help="name of the task")
        self.argparser_run.add_argument("-j",
                                        "--jobs",
                                        type=int,
                                        default=1,
                                        help="number of parallel workers")
        self.argparser_clean.add_argument("-j",
                                          "--jobs",
                                          type=int,
                                          default=1,
                                          help="number of parallel workers")
        self.argparser_run_index.add_argument(
            "index", type=int, help="index of the simulation to run")

        self.argparser_archive.add_argument(
            "-c",
            "--compression",
            type=int,
            default=9,
            help="compression level [1-9] (1: fastest, 9: best)")
        self.argparser_archive.add_argument(
            "-j",
            "--jobs",
            type=int,
            default=1,
            help="number of jobs (when pigz is available)")

    def create_working_dir(self):
        if not self.working_dir.exists():
            os.makedirs(self.working_dir)

    def list_(self, args: argparse.Namespace):
        if args.directory:
            for i, simulation in enumerate(self.simulations):
                print(i, simulation.working_dir)
        else:
            for i, simulation in enumerate(self.simulations):
                print(i, simulation.name)

    def list_tasks(self, args: argparse.Namespace):
        self.simulations[args.index].main(["list"])

    def task_info(self, args: argparse.Namespace):
        self.create_working_dir()

        with cwd.WorkingDir(self.working_dir):
            self.simulations[args.index].main(["task-info", args.name])

    def run(self, args: argparse.Namespace):
        self.logger.info("running simulation set")

        self.create_working_dir()

        tasks = []  # type: List[Callable[[], Dict[str, Any]]]
        for simulation in self.simulations:
            tasks += run_simulation(simulation)
        doit_compat.run_doit(
            tasks,
            [
                "--process=" + str(args.jobs),
                "--backend=sqlite3",
                "--db-file=:memory:",
            ],
        )

    def dry_run(self, args: argparse.Namespace):
        del args

        self.logger.info("running simulation set (dry)")

        self.create_working_dir()

        tasks = []  # type: List[Callable[[], Dict[str, Any]]]
        for simulation in self.simulations:
            tasks += dry_run_simulation(simulation)
        doit_compat.run_doit(
            tasks,
            [
                "--backend=sqlite3",
                "--db-file=:memory:",
            ],
        )

    def clean(self, args: argparse.Namespace):
        self.logger.info("cleaning simulation set")

        self.create_working_dir()

        tasks = []  # type: List[Callable[[], Dict[str, Any]]]
        for simulation in self.simulations:
            tasks += clean_simulation(simulation)
        doit_compat.run_doit(
            tasks,
            [
                "--process=" + str(args.jobs),
                "--backend=sqlite3",
                "--db-file=:memory:",
            ],
        )

    def run_index(self, args: argparse.Namespace):
        script_dir = Path(sys.argv[0]).parent.resolve()
        with cwd.WorkingDir(script_dir):
            self.logger.info("run simulation with index %d", args.index)
            self.simulations[args.index].main(["run"])

    def qdel(self, args: argparse.Namespace):
        del args

        if not self.working_dir.exists():
            self.logger.warning("working dir %s does not exist, do nothing",
                                self.working_dir)

        with cwd.WorkingDir(self.working_dir):
            for simulation in self.simulations:
                simulation.main(["qdel"])

    def qsub(self, args: argparse.Namespace):
        self.logger.info("submitting simulation set to SGE scheduler")
        self.create_working_dir()

        set_dir = self.working_dir.resolve()
        script_path = Path(sys.argv[0]).resolve()

        for index, simulation in enumerate(self.simulations):
            simulation_dir = set_dir / simulation.working_dir
            if not simulation_dir.exists():
                os.makedirs(simulation_dir)
            with cwd.WorkingDir(simulation_dir):
                sge.submit(" ".join([
                    sys.executable,
                    str(script_path), "run-index",
                    str(index)
                ]),
                           args,
                           sge_dir=script_path.parent,
                           job_name=simulation.name)

    def qsub_array(self, args: argparse.Namespace):
        self.logger.info(
            "submitting simulation set as an array to SGE scheduler")
        self.create_working_dir()

        script_path = Path(sys.argv[0]).resolve()
        command = " ".join([sys.executable, str(script_path), "run-index"])

        with cwd.WorkingDir(self.working_dir):
            sge.submit_array(command,
                             len(self.simulations),
                             args,
                             sge_dir=script_path.parent,
                             job_name=self.name)

    def archive(self, args: argparse.Namespace):
        self.create_working_dir()
        with cwd.WorkingDir(self.working_dir.resolve().parent):
            util.compress_folder(self.working_dir.name,
                                 compression=args.compression,
                                 jobs=args.jobs)

    def main(self, argv: List[str]):
        if argv is None:
            argv = sys.argv[1:]

        args = self.argparser.parse_args(argv)

        if args.subcommand is None:
            self.logger.error("No subcommand specified!")
            print()
            self.argparser.print_help(sys.stderr)
            exit(1)

        subcommand_map = {
            "archive": self.archive,
            "clean": self.clean,
            "dry-run": self.dry_run,
            "list": self.list_,
            "list-tasks": self.list_tasks,
            "qdel": self.qdel,
            "qsub": self.qsub,
            "run": self.run,
            "run-index": self.run_index,
            "task-info": self.task_info,
        }

        subcommand_map[args.subcommand](args)
