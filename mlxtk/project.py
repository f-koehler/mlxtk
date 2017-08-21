from __future__ import print_function
from builtins import input

import argparse
import os
import shutil
import subprocess
import sys

from mlxtk import log
from mlxtk import sge
from mlxtk.task.operator import OperatorCreationTask
from mlxtk.task.propagate import PropagationTask
from mlxtk.task.wavefunction import WaveFunctionCreationTask
from mlxtk.task.copy_wavefunction import CopyWavefunctionTask


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
        self.tasks.append(CopyWavefunctionTask(self, name, path))

    def create_operator(self, name, func, func_args=[]):
        self.tasks.append(OperatorCreationTask(self, name, func, func_args))

    def create_wavefunction(self, name, func, func_args=[]):
        self.tasks.append(
            WaveFunctionCreationTask(self, name, func, func_args))

    def get_logger(self, name):
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
        return os.path.join(self.root_dir, "hashes")

    def get_operator_dir(self):
        return os.path.join(self.root_dir, "operators")

    def get_wave_function_dir(self):
        return os.path.join(self.root_dir, "wave_functions")

    def improved_relax(self, initial, final, hamiltonian, **kwargs):
        kwargs["improved_relax"] = True
        kwargs["logger"] = self.get_logger("improved_relax")
        self.propagate(initial, final, hamiltonian, **kwargs)

    def main(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(title="subcommands")

        parser_clean = subparsers.add_parser("clean")
        parser_clean.set_defaults(func=self.action_clean)

        parser_ls = subparsers.add_parser("ls")
        parser_ls.set_defaults(func=self.action_ls)

        parser_qsub = subparsers.add_parser("qsub")
        parser_qsub.set_defaults(func=self.action_qsub)
        sge.add_parser_arguments(parser_qsub)

        parser_run = subparsers.add_parser("run")
        parser_run.set_defaults(func=self.action_run)

        args = parser.parse_args()
        args.func(args)

    def propagate(self, initial, final, hamiltonian, **kwargs):
        self.tasks.append(
            PropagationTask(self, initial, final, hamiltonian, **kwargs))

    def relax(self, initial, final, hamiltonian, **kwargs):
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
