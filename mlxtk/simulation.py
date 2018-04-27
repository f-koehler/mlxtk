import argparse
import json
import os
import pickle
import sys

import h5py

from . import cwd
from . import log
from . import sge
from . import task
from .inout.hdf5 import IncompleteHDF5, HDF5Error


class Simulation(object):
    def __init__(self, name, **kwargs):
        self.name = name
        self.cwd = kwargs.get("cwd", name)
        self.tasks = []
        self.parameters = kwargs.get("parameters", None)
        self.logger = log.get_logger(__name__)

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

    def compute_variance(self, propagation, operator, operator_squared,
                         **kwargs):
        self.tasks.append(
            task.ComputeVarianceTask(propagation, operator, operator_squared,
                                     **kwargs))
        return self.tasks[-1]

    def compute_basis_projection(self, wave_function, basis, **kwargs):
        self.tasks.append(
            task.BasisProjectionTask(wave_function, basis, **kwargs))
        return self.tasks[-1]

    def compute_td_basis_projection(self, propagation, basis, **kwargs):
        self.tasks.append(
            task.TDBasisProjectionTask(propagation, basis, **kwargs))
        return self.tasks[-1]

    def propagate(self, wave_function, operator, **kwargs):
        if "name" in kwargs:
            name = kwargs["name"]
            kwargs.pop("name")
        else:
            name = "propagation_" + str(self.propagation_counter)
            self.propagation_counter += 1

        self.tasks.append(
            task.PropagationTask(name, wave_function, operator, **kwargs))
        return self.tasks[-1]

    def relax(self, wave_function, operator, **kwargs):
        if "name" in kwargs:
            name = kwargs["name"]
            kwargs.pop("name")
        else:
            name = "relaxation_" + str(self.relaxation_counter)
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

        if os.path.exists("timings.pickle"):
            timings = pickle.load("timings.pickle")
        else:
            timings = {}

        for tsk in self.tasks:
            tsk.parameters = self.parameters
            result = tsk.run()
            if result["executed"]:
                timings[tsk.name] = result["time"]

        with open("timings.pickle", "wb") as fhandle:
            pickle.dump(timings, fhandle)

        with open("timings.json", "w") as fhandle:
            json.dump(timings, fhandle)

        cwd.go_back()

    def dry_run(self):
        self.logger.info("enter simulation %s", self.name)
        if self.parameters is not None:
            self.logger.info("parameters: %s", str(self.parameters))

        if not os.path.exists(self.cwd):
            os.makedirs(self.cwd)

        state_dir = os.path.join(self.cwd, "states")
        if not os.path.exists(state_dir):
            os.makedirs(state_dir)

        cwd.change_dir(self.cwd)

        tasks_to_run = []
        for tsk in self.tasks:
            tsk.parameters = self.parameters
            if not tsk.is_up_to_date():
                self.logger.info("task \"%s\" would be run", tsk.name)
                tasks_to_run.append(tsk.name)
            else:
                self.logger.info("task \"%s\" is up-to-date", tsk.name)

        self.logger.info("%d tasks would be run", len(tasks_to_run))

        cwd.go_back()
        return tasks_to_run

    def run_task(self, name):
        for t in self.tasks:
            if t.name == name:
                tsk = t
                break
        else:
            raise ValueError("task with name \"" + name + "\" does not exist")

        self.logger.info("execute only task \"" + name + "\"")
        cwd.change_dir(self.cwd)
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
                except IncompleteHDF5 as e:
                    self.logger.error(
                        "dataset is incomplete, adding data of task \"%s\" failed with exception:",
                        tsk.name)
                    self.logger.error(str(e))
                    break

        cwd.go_back()

        if opened_file:
            group.close()

    def list_tasks(self):
        return [tsk.name for tsk in self.tasks]

    def main(self):
        self.logger.info("start simulation")

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="subcommand")

        subparsers.add_parser("run")
        parser_qsub = subparsers.add_parser("qsub")

        sge.add_parser_arguments(parser_qsub)

        args = parser.parse_args()

        if args.subcommand == "run":
            self.run()
        elif args.subcommand == "qsub":
            self.qsub(args)
