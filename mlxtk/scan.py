import argparse
import copy
import datetime
import os
import pickle
import shutil
import sys

import h5py
import numpy

from mlxtk import log
from mlxtk import sge
from mlxtk.inout import InOutError
from mlxtk.parameters import ParameterTable


class ParameterScan(object):
    """Scan over ranges of parameters

    Args:
        name (str): name for the parameter scan
        parameters (mlxtk.parameters.Parameters): parameters of this scan (the
            values are takes as initial values)
        simulation_generator: function that returns a simulation given a
            parameter set

    Attributes:
        name (str): name for the parameter scan
        simulation_generator: function that returns a simulation given a
            parameter set
        scan_indices (list): list of the indices of all simulations
        simulations (list): a list of all currently generated simulations
        logger: the logger used by parameter scans
    """

    def __init__(self, name, parameters, simulation_generator, **kwargs):
        self.name = name
        self.simulation_generator = simulation_generator
        self.cwd = kwargs.get("cwd", name)

        self.table = ParameterTable(parameters)
        self.scan_indices = None

        self.simulations = []

        self.logger = log.getLogger("Scan")

    def set_values(self, name, values):
        """Specify the values a certain parameter can take

        The values will be converted to a list

        Args:
            name (str): name of the parameter
            values: parameter values as some kind of iterable
        """
        self.table.set_values(name, values)

    def check_parameters(self):
        self.logger.info("check stored parameter table if present")

        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        shelf_dir = os.path.join(self.cwd, "shelf_" + time_stamp)

        def create_shelf_dir():
            self.logger.info("create shelf directory %s", shelf_dir)
            if not os.path.exists(shelf_dir):
                os.makedirs(shelf_dir)

        def shelve(i):
            src = os.path.join(self.cwd, "sim_" + str(i))
            dst = os.path.join(shelf_dir, "sim_" + str(i))
            if os.path.exists(src):
                self.logger.info("shelve %s -> %s", src, dst)
                shutil.move(src, dst)

        table_path = os.path.join(self.cwd, "parameters.pickle")
        if not os.path.exists(table_path):
            self.logger.info(
                "no stored parameter table exists, no action required")
            return

        stored_table = ParameterTable.load(table_path)
        difference = self.table.compare(stored_table)

        # case of completely incompatible parameter tables
        if difference is None:
            self.logger.warn("parameter table incompatible, shelve everything")
            create_shelf_dir()
            for i in enumerate(stored_table):
                shelve(i)
            return

        self.logger.info("%d simulations are not present in the scan anymore",
                         len(difference["missing_rows"]))
        self.logger.info("%d simulations have to be moved",
                         len(difference["moved_rows"]))

        # shelve results contained only in the stored simulation
        missing = difference["missing_rows"]
        if missing:
            self.logger.info(
                "found extra simulations, they will be moved to the shelf")
            for miss in missing:
                index = stored_table.get_index(miss)
                shelve(index)

        # move simulations to their new isdex
        # first all these simulations are moved to a temporary dir
        tmp_dir = os.path.join(self.cwd, "tmp")
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        for _, index_from in difference["moved_rows"]:
            src = os.path.join(self.cwd, "sim_" + str(index_from))
            dst = os.path.join(tmp_dir, "sim_" + str(index_from))
            self.logger.info("move to tmp %s -> %s", src, dst)
            shutil.move(src, dst)

        # then each simulation is moved to its new place
        for index_to, index_from in difference["moved_rows"]:
            src = os.path.join(tmp_dir, "sim_" + str(index_from))
            dst = os.path.join(self.cwd, "sim_" + str(index_to))
            self.logger.info("move %s -> %s", src, dst)
            shutil.move(src, dst)

        self.logger.info("remove tmp dir %s", tmp_dir)
        shutil.rmtree(tmp_dir)

    def generate_simulations(self):
        self.simulations = []

        for values in self.table.table:
            parameters = self.table.create_parameters(values)
            simulation = self.simulation_generator(parameters)
            index = self.table.get_index(values)
            simulation.name = "sim_{}".format(index)
            simulation.cwd = "sim_{}".format(index)
            simulation.parameters = parameters

            if not simulation.is_up_to_date():
                self.simulations.append(simulation)

    def generate_simulation(self, index):
        parameters = self.table.create_parameters(self.table.table[index])
        simulation = self.simulation_generator(parameters)
        simulation.name = "sim_{}".format(index)
        simulation.cwd = "sim_{}".format(index)
        simulation.parameters = copy.copy(parameters)

        self.simulations = [simulation]

    def run(self):
        self.generate_simulations()

        if not os.path.exists(self.cwd):
            os.makedirs(self.cwd)

        self.check_parameters()

        self.table.dump(os.path.join(self.cwd, "parameters.pickle"))

        if not self.simulations:
            self.logger.info("all simulations are up-to-date")
            return

        olddir = os.getcwd()
        os.chdir(self.cwd)

        self.logger.info("found %d simulations that are not up-to-date",
                         len(self.simulations))

        for simulation in self.simulations:
            simulation.run()

        os.chdir(olddir)

    def run_index(self, index):
        self.generate_simulation(index)

        if self.simulations[0].is_up_to_date():
            self.logger.info("simulation %d is already up-to-date", index)
            return

        self.logger.info("run simulation with index %d", index)
        self.simulations[0].run()

    def qsub(self, args):
        self.generate_simulations()

        if not os.path.exists(self.cwd):
            os.makedirs(self.cwd)

        self.check_parameters()

        self.table.dump(os.path.join(self.cwd, "parameters.pickle"))

        if not self.simulations:
            self.logger.info("all simulations are up-to-date")
            return

        script_path = os.path.abspath(sys.argv[0])

        olddir = os.getcwd()
        os.chdir(self.cwd)

        for simulation in self.simulations:
            index = self.table.get_index(simulation.parameters.to_tuple())

            cmd = " ".join([
                "python",
                os.path.relpath(script_path), "--index",
                str(index), "run-index"
            ])
            job_file = "job_{}.sh".format(simulation.name)
            sge.write_job_file(job_file, simulation.name, cmd, args)

            jobid = sge.submit_job(job_file)

            sge.write_stop_script("stop_{}.sh".format(jobid), [jobid])
            sge.write_epilogue_script("epilogue_{}.sh".format(jobid), [jobid])

        os.chdir(olddir)

    def create_hdf5(self, group=None):
        self.generate_simulations()

        opened_file = group is None
        if opened_file:
            self.logger.info("create new hdf5 file")
            group = h5py.File(os.path.join(self.cwd, self.name + ".hdf5"), "w")
        else:
            self.logger.info("create hdf5 group %s", self.name)
            group = h5py.create_group(self.name)

        group.attrs["scan_parameters"] = numpy.void(
            pickle.dumps(self.get_pickle_input()))

        olddir = os.getcwd()
        os.chdir(self.cwd)

        for i, simulation in enumerate(self.simulations):
            simulation.create_hdf5(group)
            self.logger.info("%d/%d simulations processed", i + 1,
                             len(self.simulations))

        os.chdir(olddir)

        if opened_file:
            group.close()

    def main(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "action",
            metavar="action",
            type=str,
            choices=["run", "run-index", "qsub", "hdf5"],
            help="{run, run-index, qsub, hdf5}")
        parser.add_argument(
            "--index",
            type=int,
            help="index of the simulation to run when using \"run-index\"")
        sge.add_parser_arguments(parser)

        args = parser.parse_args()

        if args.action == "run":
            self.run()
        elif args.action == "run-index":
            self.run_index(args.index)
        elif args.action == "qsub":
            self.qsub(args)
        elif args.action == "hdf5":
            self.create_hdf5()
