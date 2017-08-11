from mlxtk import log
from mlxtk.task.operator import OperatorCreationTask
from mlxtk.task.propagate import PropagationTask
from mlxtk.task.wavefunction import WaveFunctionCreationTask

import argparse
import os
import shutil


def create_operator_table(*args):
    return "\n".join(args)


class Project:
    def __init__(self, name, **kwargs):
        self.name = name
        self.root_dir = kwargs.get("root_dir", os.path.join(os.getcwd(), name))
        self.operators = {}
        self.wavefunctions = {}
        self.tasks = []

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

    def create_operator(self, name, func):
        self.tasks.append(OperatorCreationTask(self, name, func))

    def create_wavefunction(self, name, func):
        self.tasks.append(WaveFunctionCreationTask(self, name, func))

    def propagate(self, initial, final, hamiltonian, **kwargs):
        self.tasks.append(
            PropagationTask(self, initial, final, hamiltonian, **kwargs))

    def relax(self, initial, final, hamiltonian, **kwargs):
        kwargs["relax"] = True
        kwargs["logger"] = log.getLogger("relax")
        self.propagate(initial, final, hamiltonian, **kwargs)

    def improved_relax(self, initial, final, hamiltonian, **kwargs):
        kwargs["improved_relax"] = True
        kwargs["logger"] = log.getLogger("improved_relax")
        self.propagate(initial, final, hamiltonian, **kwargs)

    def exact_diag(self, initial, hamiltonian, **kwargs):
        kwargs["exact_diag"] = True
        kwargs["logger"] = log.getLogger("exact_diag")
        self.propagate(initial, "exact_diag_" + initial, hamiltonian, **kwargs)

    def main(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(title="subcommands")

        def run():
            for task in self.tasks:
                task.execute()
            exit(0)

        parser_run = subparsers.add_parser("run")
        parser_run.set_defaults(func=run)

        def ls():
            if self.operators:
                print("Operators:")
                for op in self.operators:
                    print("\t" + op)

            if self.wavefunctions:
                print("Wave Functions:")
                for wfn in self.wavefunctions:
                    print("\t" + wfn)
            exit(0)

        parser_ls = subparsers.add_parser("ls")
        parser_ls.set_defaults(func=ls)

        def clean():
            choice = raw_input(
                "Remove \"{}\"? (y/n) ".format(self.root_dir)).lower()
            if choice == "y":
                shutil.rmtree(self.root_dir)
            exit(0)

        parser_clean = subparsers.add_parser("clean")
        parser_clean.set_defaults(func=clean)

        args = parser.parse_args()
        args.func()
