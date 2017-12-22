import argparse
import json
import os
import pickle
import sys

import h5py

from mlxtk import task
from mlxtk import sge
from mlxtk import log
from .inout.hdf5 import HDF5Error
from . import cwd


class Simulation(object):
    def __init__(self, name, **kwargs):
        self.name = name
        self.cwd = kwargs.get("cwd", name)
        self.tasks = []
        self.parameters = kwargs.get("parameters", None)
        self.logger = log.getLogger("Simulation")

        self.propagation_counter = 0
        self.relaxation_counter = 0

    def create_operator(self, name, function, **kwargs):
        self.tasks.append(task.OperatorCreationTask(name, function, **kwargs))
        return self.tasks[-1]

    def create_wave_function(self, name, function, **kwargs):
        self.tasks.append(
            task.WaveFunctionCreationTask(name, function, **kwargs))
        return self.tasks[-1]

    def compute_expval(self, propagation, operator, **kwargs):
        self.tasks.append(
            task.ExpectationValueTask(propagation, operator, **kwargs))
        return self.tasks[-1]

    def propagate(self, wave_function, operator, **kwargs):
        if "name" in kwargs:
            name = kwargs["name"]
            kwargs.pop("name")
        else:
            name = "propagation{}".format(self.propagation_counter)
            self.propagation_counter += 1

        self.tasks.append(
            task.PropagationTask(name, wave_function, operator, **kwargs))
        return self.tasks[-1]

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

        return self.tasks[-1]

    def is_up_to_date(self):
        if not os.path.exists(self.cwd):
            return False

        state_dir = os.path.join(self.cwd, "states")
        if not os.path.exists(state_dir):
            return False

        cwd.change_dir(self.cwd)

        for tsk in self.tasks:
            tsk.parameters = self.parameters
            if not tsk.is_up_to_date():
                cwd.go_back()
                return False

        cwd.go_back()

        return True

    def run(self):
        self.logger.info("enter simulation %s", self.name)
        if self.parameters is not None:
            self.logger.info("parameters: %s", str(self.parameters))
        if not os.path.exists(self.cwd):
            os.makedirs(self.cwd)

        state_dir = os.path.join(self.cwd, "states")
        if not os.path.exists(state_dir):
            os.makedirs(state_dir)

        cwd.change_dir(self.cwd)

        parameter_dict = {
            name: self.parameters[name]
            for name in self.parameters.names
        }
        with open("parameters.pickle", "wb") as fhandle:
            pickle.dump(parameter_dict, fhandle)

        with open("parameters.json", "w") as fhandle:
            json.dump(parameter_dict, fhandle)

        for tsk in self.tasks:
            tsk.parameters = self.parameters
            tsk.run()

        cwd.go_back()

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

    def create_hdf5(self, group=None):
        if not os.path.exists(self.cwd):
            raise HDF5Error("simulation dir is not present")

        opened_file = group is None
        if opened_file:
            self.logger.info("create new hdf5 file")
            group = h5py.File(self.name + ".hdf5")
        else:
            group = group.create_group(self.name)

        cwd.change_dir(self.cwd)

        for tsk in self.tasks:
            if getattr(tsk, "create_hdf5", None) is not None:
                try:
                    tsk.create_hdf5(group)
                except HDF5Error as e:
                    self.logger.error(
                        "failed to create hdf5 group for task \"%s\"",
                        tsk.name)
                    self.logger.error("exception: %s", str(e))
                    self.logger.error("the data is incomplete")

        cwd.go_back()

        if opened_file:
            group.close()

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
