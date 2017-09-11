from __future__ import print_function
from builtins import input

import argparse
import os
import shutil
import sys

from mlxtk import log
from mlxtk import sge
import mlxtk.task


def create_operator_table(*args):
    return "\n".join(args)


class Project(object):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, name, **kwargs):
        self.name = name
        self.root_dir = kwargs.get("root_dir", os.path.join(os.getcwd(), name))
        self.operators = {}
        self.wavefunctions = {}
        self.psis = {}
        self.tasks = []

        self.loggers = []
        self.log_file_handler = None
        self.logger = kwargs.get("logger", self.get_logger("project"))

    def action_clean(self, args):
        """Delete all project output

        Args:
            args: additional command line arguments
        """
        if os.path.exists(self.root_dir):
            choice = input(
                "Remove \"{}\"? (y/n) ".format(self.root_dir)).lower()
            if choice == "y":
                shutil.rmtree(self.root_dir)
        return 0

    def action_ls(self, args):
        for task in self.tasks:
            task.set_project_targets()

        if self.operators:
            print("Operators:")
            for operator in self.operators:
                print("\t" + operator)

        print("")

        if self.wavefunctions:
            print("Wave Functions:")
            for wfn in self.wavefunctions:
                print("\t" + wfn)

        print("")
        if self.psis:
            print("Wave Function Dynamics (Psi Files):")
            for psi in self.psis:
                print("\t" + psi)

        return 0

    def action_plot(self, args):
        for task in self.tasks:
            func = getattr(task, "plot", None)
            if not callable(func):
                continue
            func()

    def action_qsub(self, args):
        # create root directory
        self._create_root_directory()

        # set up logging
        self.log_file_handler = log.FileHandler(
            os.path.join(self.root_dir, "log"))
        for logger in self.loggers:
            logger.addHandler(self.log_file_handler)
        self.log_file_handler.setFormatter(log.formatter)
        self.logger = self.get_logger("project")

        self.logger.info("write SGE job file")
        job_file_path = os.path.join(self.root_dir, "job.sh")
        job_cmd = " ".join(["python", sys.argv[0], "run"])
        sge.write_job_file(job_file_path, self.name, job_cmd, args)

        self.logger.info("submit job")
        jobid = sge.submit_job(job_file_path)

        self.logger.info("write epilogue script")
        epilogue_script_path = os.path.join(self.root_dir, "epilogue.sh")
        sge.write_epilogue_script(epilogue_script_path, jobid)

        self.logger.info("write stop script")
        stop_script_path = os.path.join(self.root_dir, "stop.sh")
        sge.write_stop_script(stop_script_path, [jobid])

        # reset loggers and exit
        self._reset_loggers()
        return 0

    def action_run(self, args):
        # create root directory
        self._create_root_directory()

        # set up logging
        self.log_file_handler = log.FileHandler(
            os.path.join(self.root_dir, "log"))
        for logger in self.loggers:
            logger.addHandler(self.log_file_handler)
        self.log_file_handler.setFormatter(log.formatter)
        self.logger = self.get_logger("project")

        self.logger.info("run project \"%s\"", self.name)
        self.logger.info("root_dir: \"%s\"", self.root_dir)
        self._create_directories()

        # run tasks
        for task in self.tasks:
            task.execute()
        self.logger.info("done with project \"%s\"", self.name)

        # reset loggers and exit
        self._reset_loggers()
        return 0

    def copy_wave_function(self, name, path):
        self.tasks.append(mlxtk.task.CopyWavefunctionTask(self, name, path))

    def create_operator(self, name, func, func_args=[]):
        """Add an operator creation task to the project

        Args:
            name (str): Name of the operator.
            func: A callable that returns the Operator/Operatorb object.
            func_args (list): Any extra arguments func might need.
        """
        self.tasks.append(
            mlxtk.task.OperatorCreationTask(self, name, func, func_args))

    def create_wavefunction(self, name, func, func_args=[]):
        """Add a wave function creation task to the project

        Args:
            name (str): Name of the wave function
            func: A callable that returns the Wavefunction object
            func_args: (list): Any extra arguments func might need
        """
        self.tasks.append(
            mlxtk.task.WaveFunctionCreationTask(self, name, func, func_args))

    def get_logger(self, name):
        """Get the logging handler with the given name

        By using this function instead of :py:func:`logging.getLogger` it is
        ensured that the logger is associated with the file handler of the
        project.

        Args:
            name (str): Name of the logger

        Returns:
            logging.Logger: The requested logger
        """
        logger = log.getLogger(name)
        if self.log_file_handler:
            logger.addHandler(self.log_file_handler)
        self.loggers.append(logger)
        return logger

    def exact_diag(self, initial, hamiltonian, **kwargs):
        kwargs["exact_diag"] = True
        kwargs["logger"] = self.get_logger("exact_diag")
        self.propagate(initial, "exact_diag_" + initial, hamiltonian, **kwargs)

    def get_hash_dir(self):
        """Get the path of the hash directory

        Returns:
            str: Full path of the hash directory
        """
        return os.path.join(self.root_dir, "hashes")

    def get_operator_dir(self):
        return os.path.join(self.root_dir, "operators")

    def get_wave_function_dir(self):
        return os.path.join(self.root_dir, "wave_functions")

    def improved_relax(self, initial, final, hamiltonian, **kwargs):
        """Add an improved relaxation task to the project

        Args:
            initial (str): Name of the initial wave function
            final (str): Name of the final wave function
            hamiltonian (str): Name of the operator to use for the propagation
        """
        kwargs["improved_relax"] = True
        kwargs["logger"] = self.get_logger("improved_relax")
        self.propagate(initial, final, hamiltonian, **kwargs)

    def main(self):
        """Main function of the project

        This function handles command line arguments and executes the actions requested by the user
        """
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(
            title="subcommands",
            help=
            "The help for each subcommand can be shown by passing -h/--help to it."
        )

        parser_clean = subparsers.add_parser(
            "clean", help="delete all files generated by the project")
        parser_clean.set_defaults(func=self.action_clean)

        parser_ls = subparsers.add_parser("ls")
        parser_ls.set_defaults(func=self.action_ls)

        parser_plot = subparsers.add_parser(
            "plot", help="create default plots for all tasks")
        parser_plot.set_defaults(func=self.action_plot)

        parser_qsub = subparsers.add_parser(
            "qsub", help="submit this project to the SGE batch-queuing system")
        parser_qsub.set_defaults(func=self.action_qsub)
        sge.add_parser_arguments(parser_qsub)

        parser_run = subparsers.add_parser(
            "run", help="run the project locally")
        parser_run.set_defaults(func=self.action_run)

        args = parser.parse_args()
        args.func(args)

    def propagate(self, initial, final, hamiltonian, **kwargs):
        """Add a wave function propagation task to the project

        Args:
            initial (str): Name of the initial wave function
            final (str): Name of the final wave function
            hamiltonian (str): Name of the operator to use for the propagation
        """
        self.tasks.append(
            mlxtk.task.PropagationTask(self, initial, final, hamiltonian, **
                                       kwargs))

    def relax(self, initial, final, hamiltonian, **kwargs):
        """Add a relaxation task to the project

        Args:
            initial (str): Name of the initial wave function
            final (str): Name of the final wave function
            hamiltonian (str): Name of the operator to use for the propagation
        """
        kwargs["relax"] = True
        kwargs["logger"] = self.get_logger("relax")
        self.propagate(initial, final, hamiltonian, **kwargs)

    def _create_root_directory(self):
        if not os.path.exists(self.root_dir):
            self.logger.info("create project root dir")
            os.makedirs(self.root_dir)

    def _create_directories(self):
        if not os.path.exists(self.get_hash_dir()):
            self.logger.info("create hash dir")
            os.makedirs(self.get_hash_dir())

        if not os.path.exists(self.get_operator_dir()):
            self.logger.info("create operator dir")
            os.makedirs(self.get_operator_dir())

        if not os.path.exists(self.get_wave_function_dir()):
            self.logger.info("create operator dir")
            os.makedirs(self.get_wave_function_dir())

    def _reset_loggers(self):
        if self.log_file_handler:
            for logger in self.loggers:
                logger.removeHandler(self.log_file_handler)
