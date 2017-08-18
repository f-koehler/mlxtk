from builtins import input

import argparse
import itertools
import os
import shutil

from mlxtk import log


class Parameter(object):
    def __init__(self, name, values):
        self.name = name
        self.values = values
        self.indices = [i for i in range(0, len(values))]


class ParameterScan(object):
    def __init__(self, name, project_func):
        self.name = name
        self.parameters = []
        self.project_func = project_func
        self.logger = log.getLogger("scan")

        self.initial_cwd = os.getcwd()

    def action_clean(self):
        dir = self._get_working_directory()
        if not os.path.exists(dir):
            exit(0)
        choice = input("Remove \"{}\"? (y/n) ".format(dir)).lower()
        if choice == "y":
            shutil.rmtree(dir)
        exit(0)

    def action_run(self):
        self._create_working_directory()
        self._cwd()

        for project in self.generate_projects():
            project.action_run()

        self._cwd_back()

    def action_qsub(self):
        pass

    def add_parameter(self, name, values):
        self.parameters.append(Parameter(name, values))

    def evaluate_parameters(self, indices):
        return [
            parameter.values[index]
            for index, parameter in zip(indices, self.parameters)
        ]

    def generate_project(self, indices):
        parameters = self.evaluate_parameters(indices)
        project = self.project_func(*parameters)
        project.name = self.name + "_".join([str(index) for index in indices])
        project.root_dir = os.path.join(os.getcwd(), self.name, project.name)
        return project

    def generate_projects(self):
        indices = [parameter.indices for parameter in self.parameters]
        return [
            self.generate_project(element)
            for element in itertools.product(*indices)
        ]

    def main(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(title="subcommands")

        parser_clean = subparsers.add_parser("clean")
        parser_clean.set_defaults(func=self.action_clean)

        # parser_ls = subparsers.add_parser("ls")
        # parser_ls.set_defaults(func=self.action_ls)

        # parser_qsub = subparsers.add_parser("qsub")
        # parser_qsub.set_defaults(func=self.action_qsub)

        parser_run = subparsers.add_parser("run")
        parser_run.set_defaults(func=self.action_run)

        args = parser.parse_args()
        args.func()

    def _create_working_directory(self):
        if not os.path.exists(self._get_working_directory()):
            self.logger.info("create working directory")
            os.makedirs(self._get_working_directory())

    def _cwd(self):
        self.logger.info("change working directory to scan directory")
        os.chdir(self._get_working_directory())

    def _cwd_back(self):
        self.logger.info("change back to old working directory")
        os.chdir(self.initial_cwd)

    def _get_working_directory(self):
        return os.path.join(self.initial_cwd, self.name)
