import argparse
import copy
import os
import shutil
import subprocess
import sys

import h5py
import numpy

from . import cwd
from . import date
from . import log
from . import sge
from . import tabulate
from . import util
from .inout import hdf5
from .parameters import ParameterTable


def parse_simulation_selection(string):
    tokens = string.split(",")
    indices = set()
    for token in tokens:
        if "-" in token:
            i_s = [int(i) for i in token.split("-")]
            indices = indices | set(range(min(i_s), max(i_s) + 1))
        else:
            indices.add(int(token))
    return list(indices)


class ParameterScan(object):
    """Scan over ranges of parameters

    Args:
        name (str): name for the parameter scan
        parameters (mlxtk.parameters.Parameters): parameters of this scan (the
            values are taken as initial values)
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

        self.logger = log.get_logger(__name__)

    def set_values(self, name, values):
        """Specify the values a certain parameter can take

        The values will be converted to a list

        Args:
            name (str): name of the parameter
            values: parameter values as some kind of iterable
        """
        self.table.set_values(name, values)

    def add_parameter_filter(self, filt):
        self.table.filters.append(filt)
        self.table.recalculate()

    def check_parameters(self):
        self.logger.info("check stored parameter table if present")

        timestamp = date.get_timestamp_filename()
        shelf_dir = os.path.join(self.cwd, "shelf_" + timestamp)

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

        self.logger.info(
            "%d simulations are not present in the scan anymore",
            len(difference["missing_rows"]),
        )
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

        # move simulations to their new index
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

    def check_parameters_dry_run(self):

        timestamp = date.get_timestamp_filename()
        shelf_dir = os.path.join(self.cwd, "shelf_" + timestamp)

        result = {}

        def shelve(i):
            src = os.path.join(self.cwd, "sim_" + str(i))
            dst = os.path.join(shelf_dir, "sim_" + str(i))
            if os.path.exists(src):
                self.logger.info("would shelve %s to %s", src, dst)
                result["shelf"] = result.get("shelf", [])
                result["shelf"].append(i)

        table_path = os.path.join(self.cwd, "parameters.pickle")
        if not os.path.exists(table_path):
            self.logger.info(
                "no parameter table present, no rearranging required")
            return result

        stored_table = ParameterTable.load(table_path)
        difference = self.table.compare(stored_table)

        if difference is None:
            self.logger.warn(
                "parameter table incompatible, would shelve everything")
            for i in enumerate(stored_table):
                shelve(i)
            return result

        missing = difference["missing_rows"]
        if missing:
            self.logger.info(
                "found extra simulations, they will be moved to the shelf")
            for miss in missing:
                index = stored_table.get_index(miss)
                shelve(index)

        if not difference:
            return result

        result["moves"] = []
        for index_to, index_from in difference["moved_rows"]:
            result["moves"].append((index_from, index_to))

        return result

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
        log.open_log_file(
            os.path.join(self.cwd,
                         "run_" + date.get_timestamp_filename() + ".log"))

        self.generate_simulations()

        if not os.path.exists(self.cwd):
            os.makedirs(self.cwd)

        self.check_parameters()

        self.table.dump(os.path.join(self.cwd, "parameters.pickle"))

        if not self.simulations:
            self.logger.info("all simulations are up-to-date")
            return

        cwd.change_dir(self.cwd)

        self.logger.info("found %d simulations that are not up-to-date",
                         len(self.simulations))

        for simulation in self.simulations:
            simulation.run()

        cwd.go_back()

        log.close_log_file()

    def run_task(self, name):
        log.open_log_file(os.path.join(self.cwd, "run_task_" + name + ".log"))

        self.generate_simulations()

        if not os.path.exists(self.cwd):
            os.makedirs(self.cwd)

        self.check_parameters()

        self.table.dump(os.path.join(self.cwd, "parameters.pickle"))

        if not self.simulations:
            self.logger.info("all simulations are up-to-date")
            return

        cwd.change_dir(self.cwd)

        self.logger.info("found %d simulations that are not up-to-date",
                         len(self.simulations))

        for simulation in self.simulations:
            simulation.run_task(name)

        cwd.go_back()

        log.close_log_file()

    def dry_run(self):
        self.generate_simulations()

        if not os.path.exists(self.cwd):
            os.makedirs(self.cwd)

        with open("dry_run.txt", "w") as fhandle:
            changes = self.check_parameters_dry_run()
            changes["simulations"] = {}

            if "shelve" in changes:
                fhandle.write("simulations that would be shelved:\n")
                for shelved in changes["shelve"]:
                    fhandle.write("\t" + shelved + "\n")
                fhandle.write("\n\n")
                fhandle.flush()

            if "moves" in changes:
                fhandle.write("simulations that would be moved:\n")
                for moved in changes["moves"]:
                    fhandle.write("\t{} -> {}\n".format(moved[0], moved[1]))
                fhandle.write("\n\n")
                fhandle.flush()

            cwd.change_dir(self.cwd)

            fhandle.write("simulations that would perform work:\n")
            for simulation in self.simulations:
                tasks = simulation.dry_run()
                if tasks:
                    fhandle.write("\t" + simulation.name + ":\n")
                    changes["simulations"][simulation.name] = tasks
                    for tsk in tasks:
                        fhandle.write("\t\t" + tsk + "\n")
                    fhandle.write("\n")
                fhandle.flush()

            cwd.go_back()

            for sim_name in changes["simulations"]:
                print("sim:", sim_name)
                print(" ", changes["simulations"][sim_name], "\n")

            if "moves" in changes:
                print("\nmoves:")
                for move in changes["moves"]:
                    print(" ", move[0], "->", move[1])

            if "shelve" in changes:
                print("\nshelf:")
                for shelved in changes["shelve"]:
                    print(" ", shelved)

    def run_index(self, index):
        log.open_log_file(
            os.path.join("sim_" + str(index),
                         "sim_" + date.get_timestamp_filename() + ".log"))

        self.generate_simulation(index)

        if self.simulations[0].is_up_to_date():
            self.logger.info("simulation %d is already up-to-date", index)
            return

        self.logger.info("run simulation with index %d", index)
        self.simulations[0].run()

        log.close_log_file()

    def qsub(self, args):
        log.open_log_file(
            os.path.join(self.cwd,
                         "qsub_" + date.get_timestamp_filename() + ".log"))

        self.generate_simulations()

        if args.indices is not None:
            indices = parse_simulation_selection(args.indices)
            self.simulations = [self.simulations[i] for i in indices]

        if not os.path.exists(self.cwd):
            os.makedirs(self.cwd)

        self.check_parameters()

        self.table.dump(os.path.join(self.cwd, "parameters.pickle"))

        if not self.simulations:
            self.logger.info("all simulations are up-to-date")
            return

        script_path = os.path.abspath(sys.argv[0])

        cwd.change_dir(self.cwd)

        if not os.path.exists("job"):
            os.makedirs("job")

        if not os.path.exists("epilogue"):
            os.makedirs("epilogue")

        if not os.path.exists("stop"):
            os.makedirs("stop")

        jobids = []
        jobid_pindices = []
        for simulation in self.simulations:
            index = self.table.get_index(simulation.parameters.to_tuple())

            cmd = " ".join([
                "python",
                os.path.relpath(script_path),
                "run-index",
                "--index",
                str(index),
            ])
            job_file = os.path.join("job", "{}.sh".format(simulation.name))
            sge.write_job_file(job_file, self.name + "_" + simulation.name,
                               cmd, args)

            jobid = sge.submit_job(job_file)
            jobids.append(jobid)
            jobid_pindices.append(index)

            sge.write_stop_script(
                os.path.join("epilogue", "{}.sh".format(jobid)), [jobid])
            sge.write_epilogue_script(
                os.path.join("stop", "{}.sh".format(jobid)), [jobid])

        sge.write_stop_script("stop_all.sh", jobids)

        with open("jobids.txt", "w") as fhandle:
            for id_, idx in zip(jobids, jobid_pindices):
                fhandle.write("{} -> {} \n".format(idx, id_))

        subdirs = ["sim_" + str(pindex) for pindex in jobid_pindices]
        write_sync_script("qsync.sh", jobids, subdirs)

        cwd.go_back()

        log.close_log_file()

    def submit_tmux(self, args):
        log.open_log_file(
            os.path.join(
                self.cwd,
                "submit_tmux_" + date.get_timestamp_filename() + ".log"))

        self.generate_simulations()

        if not os.path.exists(self.cwd):
            os.makedirs(self.cwd)

        self.check_parameters()

        self.table.dump(os.path.join(self.cwd, "parameters.pickle"))

        if not self.simulations:
            self.logger.info("all simulations are up-to-date")
            return

        script_path = os.path.abspath(sys.argv[0])

        cwd.change_dir(self.cwd)

        indices = [
            self.table.get_index(sim.parameters.to_tuple())
            for sim in self.simulations
        ]

        def run_simulation(sim_index):
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            cmd = [
                "python",
                os.path.relpath(script_path),
                "run-index",
                "--index",
                str(sim_index),
            ]
            subprocess.run(cmd, env=env)

        util.parallel_map(run_simulation, indices, args.jobs)

        cwd.go_back()
        log.close_log_file()

    def create_hdf5(self, group=None):
        log.open_log_file(
            os.path.join(self.cwd,
                         "hdf5" + date.get_timestamp_filename() + ".log"))

        self.generate_simulations()

        opened_file = group is None
        if opened_file:
            self.logger.info("create new hdf5 file")
            group = h5py.File(os.path.join(self.cwd, self.name + ".hdf5"), "w")
        else:
            self.logger.info("create hdf5 group %s", self.name)
            group = h5py.create_group(self.name)

        group.attrs["scan_parameters"] = numpy.void(self.table.dumps())

        cwd.change_dir(self.cwd)

        for i, simulation in enumerate(self.simulations):
            try:
                simulation.create_hdf5(group)
                self.logger.info("%d/%d simulations processed", i + 1,
                                 len(self.simulations))
            except hdf5.HDF5Error:
                self.warn(
                    "simulation %d does not exist, HDF5 file will be incomplete"
                )

        cwd.go_back()

        if opened_file:
            group.close()

        log.close_log_file()

    def print_table(self, args=None):
        self.generate_simulations()
        if args is None:
            print(self.table.format_table("orgtbl"))
        else:
            print(self.table.format_table(args.fmt))

    def print_constants(self, args=None):
        self.generate_simulations()
        if args is None:
            print(self.table.format_constants("orgtbl"))
        else:
            print(self.table.format_constants(args.fmt))

    def print_variables(self, args=None):
        self.generate_simulations()
        if args is None:
            print(self.table.format_variables("orgtbl"))
        else:
            print(self.table.format_variables(args.fmt))

    def print_summary(self):
        timestamp = date.get_timestamp_filename()
        print("\n".join([
            "#+TITLE: Parameter Scan Summary",
            "#+CREATOR: mlxtk",
            "#+DATE: " + timestamp,
            "",
        ]))
        print("\n\n* Constants\n")
        self.print_constants()
        print("\n\n* Variables\n")
        self.print_variables()
        print("\n\n* Parameter Table\n")
        self.print_table()

    def list_tasks(self):
        self.generate_simulations()
        print("tasks in sim_0:", self.simulations[0].list_tasks())

    def main(self):
        self.logger.info("start parameter scan")

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="subcommand")

        subparsers.add_parser("run")
        parser_run_index = subparsers.add_parser("run-index")
        parser_qsub = subparsers.add_parser("qsub")
        subparsers.add_parser("hdf5")
        parser_table = subparsers.add_parser("table")
        parser_constants = subparsers.add_parser("constants")
        parser_variables = subparsers.add_parser("variables")
        subparsers.add_parser("summary")
        subparsers.add_parser("list-tasks")
        parser_run_task = subparsers.add_parser("run-task")
        subparsers.add_parser("dry-run")
        parser_submit_tmux = subparsers.add_parser("submit-tmux")

        parser_run_index.add_argument(
            "--index",
            type=int,
            help="index of the simulation to run when using \"run-index\"",
        )

        parser_run_task.add_argument("task", type=str, help="")

        parser_submit_tmux.add_argument(
            "-j",
            "--jobs",
            type=int,
            default=1,
            help="number of threads to use")

        sge.add_parser_arguments(parser_qsub)
        parser_qsub.add_argument("-i", "--indices")

        tabulate.create_tabulate_argument(parser_table, ["-f", "--format"],
                                          "fmt")
        tabulate.create_tabulate_argument(parser_constants, ["-f", "--format"],
                                          "fmt")
        tabulate.create_tabulate_argument(parser_variables, ["-f", "--format"],
                                          "fmt")

        args = parser.parse_args()

        if args.subcommand == "run":
            self.run()
        elif args.subcommand == "run-index":
            self.run_index(args.index)
        elif args.subcommand == "run-task":
            self.run_task(args.task)
        elif args.subcommand == "qsub":
            self.qsub(args)
        elif args.subcommand == "hdf5":
            self.create_hdf5()
        elif args.subcommand == "table":
            self.print_table(args)
        elif args.subcommand == "constants":
            self.print_constants(args)
        elif args.subcommand == "variables":
            self.print_variables(args)
        elif args.subcommand == "summary":
            self.print_summary()
        elif args.subcommand == "list-tasks":
            self.list_tasks()
        elif args.subcommand == "dry-run":
            self.dry_run()
        elif args.subcommand == "submit-tmux":
            self.submit_tmux(args)
        else:
            raise ValueError("Invalid subcommand \"" + args.subcommand + "\"")


def write_sync_script(path, jobids, subdirs):
    script = ["#!/bin/bash", "set -eu -o pipefail", ""]

    for jid, subdir in zip(jobids, subdirs):
        script.append(
            "working_dir=$(qstat -j {} | grep -Po \"(?<=^cwd:).+$\" | sed -e \"s/^[ \\t]*//\")/{}".
            format(jid, subdir))
        script.append("retval=$?")
        script.append("if [ $retval -eq 0 ]; then")
        script.append("    state=$(qstat | grep \"^{}\"".format(jid) +
                      " | awk \'{ print $5 }\')")
        script.append("    if [ \"${state}\" == \"r\" ]; then")
        script.append(
            "        node=$(qstat | grep \"^{}\" | cut -d \"@\" -f2 | cut -d \".\" -f1)".
            format(jid))
        script.append("        echo \"syncing job {} ({})\"".format(
            jid, subdir))
        script.append("        mkdir -p \"${working_dir}_tmp\"")
        script.append(
            "        sshpass -v -p ${SSH_PASSWORD} scp -v -r ${node}:\"${working_dir}/\"* \"${working_dir}_tmp/\""
        )
        script.append(
            "        cp -r \"${working_dir}_tmp/\"* \"${working_dir}\"")
        script.append("        rm -rf \"${working_dir}_tmp\"")
        script.append("        echo \"finished syncing job {} ({})\"".format(
            jid, subdir))
        script.append("    else")
        script.append(
            "        echo \"job {} ({}) is not running, skipping\"".format(
                jid, subdir))
        script.append("    fi")
        script.append("fi")
        script.append("echo \"\"")
        script.append("")

    with open(path, "w") as fhandle:
        fhandle.write("\n".join(script))
        sge.mark_file_executable(path)
