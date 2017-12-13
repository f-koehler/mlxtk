import argparse
import copy
import datetime
import itertools
import os
import pickle
import shutil
import sys

import h5py
import numpy
from tabulate import tabulate

from mlxtk import log
from mlxtk import sge
from mlxtk.inout import InOutError


class ParameterScan(object):
    """Scan over ranges of parameters

    Args:
        name (str): name for the parameter scan
        parameters (mlxtk.parameters.Parameters): parameters of this scan (the values are takes as initial values)
        simulation_generator: function that returns a simulation given a parameter set

    Attributes:
        name (str): name for the parameter scan
        simulation_generator: function that returns a simulation given a parameter set
        parameters (mlxtk.parameters.Parameters): the set of parameters
        parameter_values (list): a list of the parameter value lists
        parameter_indices (list): a list of index ranges for the individual parameters
        table_indices (list): list of all parameter multi indices (as lists)
        table_values (list): list of all parameter sets (as lists)
        scan_indices (list): list of the indices of all simulations
        filters (list): filters that will be applied to remove parameter configurations from the cartesian product
        simulations (list): a list of all currently generated simulations
        logger: the logger used by parameter scans
    """

    def __init__(self, name, parameters, simulation_generator, **kwargs):
        self.name = name
        self.simulation_generator = simulation_generator
        self.cwd = kwargs.get("cwd", name)

        self.parameters = parameters
        self.parameter_values = [[parameters[name]]
                                 for name in parameters.parameter_names]
        self.parameter_indices = [[0] for name in parameters.parameter_names]

        self.table_indices = None
        self.table_values = None
        self.scan_indices = None
        self.filters = []

        self.simulations = []

        self.logger = log.getLogger("Scan")

    def set_values(self, name, values):
        """Specify the values a certain parameter can take

        The values will be converted to a list

        Args:
            name (str): name of the parameter
            values: parameter values as some kind of iterable
        """
        index = self.parameters.parameter_names.index(name)
        self.parameter_values[index] = [val for val in values]
        self.parameter_indices[index] = [i for i, _ in enumerate(values)]

        self.table_indices = None
        self.table_values = None

    def init_tables(self):
        self.logger.info("initialize scan table")
        self.table_indices = list(itertools.product(*self.parameter_indices))
        self.table_values = list(itertools.product(*self.parameter_values))
        self.logger.info("table has %d entries", len(self.table_indices))

        for i in range(len(self.table_indices)):
            self.table_indices[i] = list(self.table_indices[i])

        for i in range(len(self.table_values)):
            self.table_values[i] = list(self.table_values[i])

        for i, filt in enumerate(self.filters):
            self.logger.info("apply filter #%d", i)
            new_table_indices = []
            new_table_values = []
            for indices, values in zip(self.table_indices, self.table_values):
                self.parameters.set_all(*values)
                if filt(self.parameters):
                    new_table_indices.append(indices)
                    new_table_values.append(values)

            self.logger.info("removed %d entries",
                             len(self.table_indices - len(new_table_indices)))
            self.table_indices = new_table_indices
            self.table_values = new_table_values

        self.scan_indices = list(range(len(self.table_indices)))

    def get_parameter_names(self):
        return [name for name in self.parameters.parameter_names]

    def get_pickle_input(self):
        self.init_tables()

        parameter_names = self.get_parameter_names()

        return {
            "names": parameter_names,
            "indices": self.parameter_indices,
            "values": self.parameter_values,
            "table_indices": self.table_indices,
            "table_values": self.table_values
        }

    def write_table(self):
        self.init_tables()

        parameter_names = self.get_parameter_names()

        tables = self.get_pickle_input()

        # write machine readable version
        pickle_file = os.path.join(self.cwd, "parameters.pickle")
        self.logger.info("pickle scan parameters (\"%s\")", pickle_file)
        with open(pickle_file, "wb") as fhandle:
            pickle.dump(tables, fhandle)

        # determine constant parameters
        constants = []
        variables = []
        for i, values in enumerate(self.parameter_values):
            if len(values) > 1:
                variables.append(i)
            else:
                constants.append(i)

        def remove_constants(lst):
            return [el for i, el in enumerate(lst) if i not in constants]

        # write human readable version
        self.logger.info("write scan overview as Emacs org-mode document")
        table_indices = tabulate(
            [[self.scan_indices[i]] + remove_constants(indices)
             for i, indices in enumerate(self.table_indices)],
            headers=["Index"] +
            remove_constants(self.parameters.parameter_names),
            tablefmt="orgtbl")
        table_values = tabulate(
            [[self.scan_indices[i]] + remove_constants(values)
             for i, values in enumerate(self.table_values)],
            headers=["Index"] +
            remove_constants(self.parameters.parameter_names),
            tablefmt="orgtbl")

        with open(os.path.join(self.cwd, "parameters.org"), "w") as fhandle:
            fhandle.write("* Parameter Scan Overview\n\n")

            fhandle.write("** Constants\n\n")
            for index in constants:
                fhandle.write("   {} = {}\n".format(
                    parameter_names[index], self.parameter_values[index][0]))
            fhandle.write("\n\n")

            fhandle.write("** Value Table\n\n")
            fhandle.write(table_values)
            fhandle.write("\n\n\n")

            fhandle.write("** Index Table\n\n")
            fhandle.write(table_indices)
            fhandle.write("\n\n\n")

    def check_parameters(self):
        self.init_tables()

        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        shelf_dir = os.path.join(self.cwd, "shelf_" + time_stamp)

        # generate current pickle
        current_pickle = self.get_pickle_input()

        # read stored pickle if available
        pickle_file = os.path.join(self.cwd, "parameters.pickle")
        if not os.path.exists(pickle_file):
            # new run of scan -> no incompatibility of parameters
            self.logger.info("pickle file does not exist, no action required")
            return

        with open(pickle_file, "rb") as fhandle:
            stored_pickle = pickle.load(fhandle)

        # check if nothing changed at all
        if stored_pickle == current_pickle:
            self.logger.info(
                "parameter pickle is unchanged, no action required")
            return

        # check if names/number of parameters changed
        if stored_pickle["names"] != current_pickle["names"]:
            # back up all generated data
            self.logger.warn("number/names/order of parameters changed")
            self.logger.debug("old: %s", str(stored_pickle["names"]))
            self.logger.debug("new: %s", str(current_pickle["names"]))
            if not os.path.exists(shelf_dir):
                os.makedirs(shelf_dir)
            for index in self.scan_indices:
                src = os.path.join(self.cwd, "sim_" + str(index))
                if not os.path.exists(src):
                    continue
                dst = os.path.join(shelf_dir, "sim_" + str(index))
                self.logger.debug("move %s -> %s", src, dst)
                shutil.move(src, dst)
            shutil.copy2(
                os.path.join(self.cwd, "parameters.pickle"),
                os.path.join(shelf_dir, "parameters.pickle"))
            shutil.copy2(
                os.path.join(self.cwd, "parameters.org"),
                os.path.join(shelf_dir, "parameters.org"))
            return

        # go through all simulations and check if they are still contained in
        # the scan and rename them accordingly
        # move all other data to the shelf
        stored_scan_indices = list(range(len(stored_pickle["table_values"])))
        for index in stored_scan_indices:
            sim = "sim_" + str(index)
            src = os.path.join(self.cwd, sim)
            if not os.path.exists(src):
                # no data for this simulation does exist
                continue

            values = stored_pickle["table_values"][index]
            if values not in current_pickle["table_values"]:
                # this set of parameters is not present in the scan anymore
                self.logger.warn(
                    "%s is not contained in the scan anymore, shelving it",
                    sim)
                dst = os.path.join(shelf_dir, sim)
                self.logger.debug("move %s -> %s", src, dst)
                shutil.move(src, dst)
                continue

            current_index = current_pickle["table_values"].index(values)
            if current_index == index:
                # this set of parameters still has the same index
                continue

            # the index of the parameter set changed
            dst = os.path.join(self.cwd, sim)
            self.logger.warn("%s has now a different index, moving it", sim)
            self.logger.debug("move %s -> %s", src, dst)
            shutil.move(src, dst)

    def generate_simulations(self):
        self.init_tables()

        self.simulations = []

        for values in self.table_values:
            self.parameters.set_all(*values)
            simulation = self.simulation_generator(self.parameters)
            index = self.table_values.index(values)
            simulation.name = "sim_{}".format(index)
            simulation.cwd = "sim_{}".format(index)
            simulation.parameters = copy.copy(self.parameters)

            if not simulation.is_up_to_date():
                self.simulations.append(simulation)

    def generate_simulation(self, index):
        self.init_tables()

        self.parameters.set_all(*self.table_values[index])
        simulation = self.simulation_generator(self.parameters)
        simulation.name = "sim_{}".format(index)
        simulation.cwd = "sim_{}".format(index)
        simulation.parameters = copy.copy(self.parameters)

        self.simulations = [simulation]

    def run(self):
        self.generate_simulations()

        if not os.path.exists(self.cwd):
            os.makedirs(self.cwd)

        self.check_parameters()

        self.write_table()

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

        self.write_table()

        if not self.simulations:
            self.logger.info("all simulations are up-to-date")
            return

        script_path = os.path.abspath(sys.argv[0])

        olddir = os.getcwd()
        os.chdir(self.cwd)

        for simulation in self.simulations:
            index = self.table_values.index(simulation.parameters.to_list())

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
            try:
                simulation.create_hdf5(group)
            except InOutError:
                self.logger.error(
                    "failed to create complete HDF5 group for simulation \"%s\"",
                    simulation.name)
                self.logger.error("the data is probably incomplete")
                os.chdir(olddir)
                os.chdir(self.cwd)
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
