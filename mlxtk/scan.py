import argparse
import copy
import itertools
import os
import pickle
import shutil
import sys

from tabulate import tabulate

from mlxtk import log
from mlxtk import sge


class ParameterScan(object):
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
            "table_values": self.table_values,
            "number_filter": len(self.filters)
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

    def parameters_changed(self):
        self.init_tables()

        current = self.get_pickle_input()

        pickle_file = os.path.join(self.cwd, "parameters.pickle")

        with open(pickle_file, "rb") as fhandle:
            stored = pickle.load(fhandle)

        if stored != current:
            return True

        return False

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

        if os.path.exists(os.path.join(self.cwd, "parameters.pickle")):
            if self.parameters_changed():
                # TODO: handle this properly, i.e. check which data is still
                # good
                self.logger.info(
                    "scan parameters changed, remove all the data")
                shutil.rmtree(self.cwd)
                os.makedirs(self.cwd)

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

        if os.path.exists(os.path.join(self.cwd, "parameters.pickle")):
            if self.parameters_changed():
                # TODO: check for still usable data
                self.logger.info(
                    "scan parameters changed, remove all the data")
                shutil.rmtree(self.cwd)
                os.makedirs(self.cwd)

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

    def main(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "action",
            metavar="action",
            type=str,
            choices=["run", "run-index", "qsub"],
            help="{run, run-index, qsub}")
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
