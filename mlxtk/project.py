from __future__ import print_function

import argparse
import os
import shutil
import subprocess
import sys

from mlxtk import log
from mlxtk.task.operator import OperatorCreationTask
from mlxtk.task.propagate import PropagationTask
from mlxtk.task.wavefunction import WaveFunctionCreationTask


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

    def action_clean(self):
        if not os.path.exists(self.root_dir):
            self.logger.error(
                "project dir is not present, no need to clean up")
            exit(1)
        choice = raw_input(
            "Remove \"{}\"? (y/n) ".format(self.root_dir)).lower()
        if choice == "y":
            shutil.rmtree(self.root_dir)
        exit(0)

    def action_ls(self):
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
        exit(0)

    def action_qsub(self):
        # create root directory
        self._create_root_directory()

        # set up logging
        self.log_file_handler = log.FileHandler(
            os.path.join(self.root_dir, "log"))
        for logger in self.loggers:
            logger.addHandler(self.log_file_handler)
        self.log_file_handler.setFormatter(log.formatter)
        self.logger = self.get_logger("project")

        # create job file
        # yapf: disable
        job_file = [
            "#!/bin/bash",
            "#$ -N " + self.name,
            "#$ -q quantix.q",
            "#$ -S /bin/bash",
            "#$ -cwd",
            "#$ -j y",
            "#$ -V",
            "#$ -l h_vmem=8G",
            "#$ -l h_cpu=0:10:00",
            "python " + sys.argv[0] + " run"
        ]
        # yapf: enable
        self.logger.debug("contents of job file")
        for line in job_file:
            self.logger.debug("  " + line)

        proc = subprocess.Popen(["qsub"], stdin=subprocess.PIPE)
        proc.communicate("\n".join(job_file))
        return_code = proc.wait()
        if return_code:
            raise subprocess.CalledProcessError("qsub", return_code)

        # reset loggers and exit
        self._reset_loggers()
        exit(0)

    def action_run(self):
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
        exit(0)

    def create_operator(self, name, func):
        self.tasks.append(OperatorCreationTask(self, name, func))

    def create_wavefunction(self, name, func):
        self.tasks.append(WaveFunctionCreationTask(self, name, func))

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

        parser_run = subparsers.add_parser("run")
        parser_run.set_defaults(func=self.action_run)

        args = parser.parse_args()
        args.func()

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
