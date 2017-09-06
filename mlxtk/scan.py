from builtins import input

import argparse
import itertools
import os
import shutil
import sys

from mlxtk import log
from mlxtk import sge


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

    def action_clean(self, args):
        dir = self._get_working_directory()
        if not os.path.exists(dir):
            exit(0)
        choice = input("Remove \"{}\"? (y/n) ".format(dir)).lower()
        if choice == "y":
            shutil.rmtree(dir)
        exit(0)

    def action_run(self, args):
        self._create_working_directory()
        self._cwd()

        for project in self.generate_projects():
            project.action_run(argparse.Namespace())

        self._cwd_back()

    def action_run_index(self, args):
        project = self.generate_project(
            [int(idx) for idx in args.id.split("_")])
        project.name = args.id
        project.action_run(argparse.Namespace())

    def action_qsub(self, args):
        self._create_working_directory()

        indices = [parameter.indices for parameter in self.parameters]
        counter = 0
        jobids = []
        for element in itertools.product(*indices):
            idxs = "_".join([str(idx) for idx in element])
            self.logger.info("job %d: parameter indices: %s", counter, idxs)

            args.output = os.path.join(self._get_working_directory(),
                                       idxs + ".out")

            self.logger.info("job %d: write SGE job file", counter)
            job_file_path = os.path.join(self._get_working_directory(),
                                         "job_{}.sh".format(idxs))
            output_path = os.path.join(self._get_working_directory(),
                                       idxs + ".out")
            job_cmd = " ".join(["python", sys.argv[0], "run-index", idxs])
            sge.write_job_file(
                job_file_path,
                self.name + "_" + idxs,
                job_cmd,
                args,
                output=output_path)

            # self.logger.info("job %d: submit job", counter)
            jobid = sge.submit_job(job_file_path)
            jobids.append(jobid)

            self.logger.info("write epilogue script")
            epilogue_script_path = os.path.join(self._get_working_directory(),
                                                "epilogue_{}.sh".format(idxs))
            sge.write_epilogue_script(epilogue_script_path, jobid)

            self.logger.info("write stop script")
            stop_script_path = os.path.join(self._get_working_directory(),
                                            "stop_{}.sh".format(idxs))
            sge.write_stop_script(stop_script_path, [jobid])

            counter += 1

        self.logger.info("write stop script for all jobs")
        sge.write_stop_script(
            os.path.join(self._get_working_directory(), "stop_all.sh"), jobids)

        self.logger.info("submitted %d jobs", counter)

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
        subparsers = parser.add_subparsers(
            title="subcommands",
            help=
            "The help for each subcommand can be shown by passing -h/--help to it."
        )

        parser_clean = subparsers.add_parser(
            "clean", help="delete all files create by the parameter scan")
        parser_clean.set_defaults(func=self.action_clean)

        # parser_ls = subparsers.add_parser("ls")
        # parser_ls.set_defaults(func=self.action_ls)

        # parser_plot = subparsers.add_parameter()

        parser_qsub = subparsers.add_parser(
            "qsub",
            help="submit the parameter scan to the SGE batch-queuing system")
        parser_qsub.set_defaults(func=self.action_qsub)
        sge.add_parser_arguments(parser_qsub)

        parser_run = subparsers.add_parser(
            "run", help="run parameter scan locally")
        parser_run.set_defaults(func=self.action_run)

        parser_run_id = subparsers.add_parser(
            "run-index",
            help="run project with a certain parmeter index (internal use)")
        parser_run_id.set_defaults(func=self.action_run_index)
        parser_run_id.add_argument(
            "id", type=str, help="index of the job to run")

        args = parser.parse_args()
        args.func(args)

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
        self.initial_cwd = os.getcwd()
        return os.path.join(self.initial_cwd, self.name)
