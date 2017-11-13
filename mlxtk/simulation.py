import argparse
import os
import sys

from mlxtk import task
from mlxtk import sge


class Simulation(object):
    def __init__(self, name, **kwargs):
        self.name = name
        self.cwd = kwargs.get("cwd", name)
        self.tasks = []
        self.parameters = kwargs.get("parameters", None)

        self.propagation_counter = 0
        self.relaxation_counter = 0

    def create_operator(self, name, function, **kwargs):
        self.tasks.append(task.OperatorCreationTask(name, function, **kwargs))

    def create_wave_function(self, name, function, **kwargs):
        self.tasks.append(
            task.WaveFunctionCreationTask(name, function, **kwargs))

    def propagate(self, wave_function, operator, **kwargs):
        if "name" in kwargs:
            name = kwargs["name"]
            kwargs.pop("name")
        else:
            name = "propagation{}".format(self.propagation_counter)
            self.propagation_counter += 1

        self.tasks.append(
            task.PropagationTask(name, wave_function, operator, **kwargs))

    def relax(self, wave_function, operator, **kwargs):
        if "name" in kwargs:
            name = kwargs["name"]
            kwargs.pop("name")
        else:
            name = "relaxation{}".format(self.relaxation_counter)
            self.relaxation_counter += 1

        kwargs["relax"] = True
        self.tasks.append(
            task.PropagationTask(name, wave_function, operator, **kwargs))

    def run(self):
        if not os.path.exists(self.cwd):
            os.makedirs(self.cwd)

        state_dir = os.path.join(self.cwd, "states")
        if not os.path.exists(state_dir):
            os.makedirs(state_dir)

        olddir = os.getcwd()
        os.chdir(self.cwd)

        for tsk in self.tasks:
            tsk.parameters = self.parameters
            tsk.run()

        os.chdir(olddir)

    def qsub(self, args):
        script_path = os.path.abspath(sys.argv[0])
        if not os.path.exists(self.cwd):
            os.makedirs(self.cwd)

        cmd = " ".join(["python", os.path.relpath(script_path), "run"])
        job_file = os.path.join(self.cwd, "job_{}.sh".format(self.name))
        sge.write_job_file(job_file, self.name, cmd, args)
        jobid = sge.submit_job(job_file)
        sge.write_stop_script(
            os.path.join(self.cwd, "stop_{}.sh".format(jobid)), [jobid])
        sge.write_epilogue_script(
            os.path.join(self.cwd, "epilogue_{}.sh".format(jobid)), jobid)

    def main(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "action",
            metavar="action",
            type=str,
            choices=["run", "qsub"],
            help="{run, qsub}")
        sge.add_parser_arguments(parser)

        args = parser.parse_args()

        if args.action == "run":
            self.run()
        elif args.action == "qsub":
            self.qsub(args)
