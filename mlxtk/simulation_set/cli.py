import argparse

from .. import sge


def construct_argparser(self):

    self.argparser_list = self.subparsers.add_parser("list")
    self.argparser_list_tasks = self.subparsers.add_parser("list-tasks")
    self.argparser_lockfiles = self.subparsers.add_parser("lockfiles")
    self.argparser_task_info = self.subparsers.add_parser("task-info")
    self.subparsers.add_parser("qdel")
    self.argparser_qsub = self.subparsers.add_parser("qsub")
    self.argparser_run = self.subparsers.add_parser("run")
    self.argparser_dry_run = self.subparsers.add_parser("dry-run")
    self.argparser_run_index = self.subparsers.add_parser("run-index")
    self.argparser_clean = self.subparsers.add_parser("clean")
    self.argparser_archive = self.subparsers.add_parser("archive")
    self.argparser_propagation_status = self.subparsers.add_parser(
        "propagation-status")

    sge.add_parser_arguments(self.argparser_qsub)

    self.argparser_list.add_argument("-d", "--directory", action="store_true")

    self.argparser_list_tasks.add_argument(
        "index", type=int, help="index of the simulation whose tasks to list")
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
    self.argparser_propagation_status.add_argument(
        "name",
        default="propagate",
        nargs="?",
        type=str,
        help="name of the propagation")
