from mlxtk.task.task import Task
from mlxtk.filesystem import relative_symlink

import logging
import os.path
import re
import shutil
import subprocess


class Propagation(Task):

    def __init__(self, initial_wavefunction, final_wavefunction, operator,
                 **kwargs):
        Task.__init__(self, "propagate_{}_to_{}".format(initial_wavefunction,
                                                        final_wavefunction))

        self.type = "Propagation"

        self.initial_wavefunction = initial_wavefunction
        self.final_wavefunction = final_wavefunction
        self.operator = operator

        self.dt = kwargs.get("dt", 1.)
        self.tfinal = kwargs.get("tfinal", self.dt)
        self.continuation = kwargs.get("continuation", False)

    def is_up_to_date(self):
        hash_path = self.get_hash_path()
        input_path = self.get_input_path()
        output_path = self.get_output_path()
        operator_path = self.get_operator_path()

        if not os.path.exists(input_path):
            raise RuntimeError("Wavefunction \"{}\" does not exist (\"{}\")".
                               format(self.initial_wavefunction, input_path))

        if not os.path.exists(operator_path):
            raise RuntimeError("Operator \"{}\" does not exist (\"{}\")".format(
                self.operator, operator_path))

        if not os.path.exists(hash_path):
            return False

        if self.hash() != self.read_hash_file():
            return False

        if not os.path.exists(output_path):
            return False

        cond1 = (os.path.getmtime(output_path) > os.path.getmtime(input_path))
        cond2 = (
            os.path.getmtime(output_path) > os.path.getmtime(operator_path))
        return (cond1 or cond2)

    def get_input_path(self):
        return os.path.join("wavefunctions",
                            self.initial_wavefunction + ".wfn")

    def get_output_path(self):
        return os.path.join("wavefunctions",
                            self.final_wavefunction + ".wfn")

    def get_operator_path(self):
        return os.path.join("operators", self.operator + ".op")

    def _copy_operator(self):
        dst = os.path.join(self.get_tmp_dir(), self.operator + ".op")
        logging.info("Copy Hamiltonian: %s -> %s", self.get_operator_path(),
                     dst)
        shutil.copy(self.get_operator_path(), dst)

    def _copy_initial_wavefunction(self):
        dst = os.path.join(self.get_tmp_dir(),
                           self.initial_wavefunction + ".wfn")
        logging.info("Copy initial wave function: %s -> %s",
                     self.get_input_path(), dst)
        shutil.copy(self.get_input_path(), dst)

    def _symlink_final_wavefunction(self):
        src = os.path.join(self.get_tmp_dir(), "restart")
        dst = self.get_output_path()
        logging.info("Copy initial wave function: %s -> %s", src,
                     dst)
        relative_symlink(src, dst)

    def _symlink_natpop(self):
        src = os.path.join(self.get_tmp_dir(), "natpop")
        dst = os.path.join("natural_populations",
                           self.name + ".natpop")
        logging.info("Symlink natural population: %s -> %s", src, dst)
        relative_symlink(src, dst)

    def _symlink_gpop(self):
        src = os.path.join(self.get_tmp_dir(), "gpop")
        dst = os.path.join("gpops", self.name + ".gpop")
        logging.info("Symlink gpop: %s -> %s", src, dst)
        relative_symlink(src, dst)

    def _symlink_output(self):
        src = os.path.join(self.get_tmp_dir(), "output")
        dst = os.path.join("outputs", self.name + ".out")
        logging.info("Symlink output: %s -> %s", src, dst)
        relative_symlink(src, dst)

    def _create_output_dirs(self):
        dir = os.path.join("natural_populations")
        if not os.path.exists(dir):
            logging.info("Create directory: %s", dir)
            os.mkdir(dir)

        dir = os.path.join("gpops")
        if not os.path.exists(dir):
            logging.info("Create directory: %s", dir)
            os.mkdir(dir)

        dir = os.path.join("outputs")
        if not os.path.exists(dir):
            logging.info("Create directory: %s", dir)
            os.mkdir(dir)

    def _execute(self):
        logging.info("Execute propagation procedure: %s -> %s",
                     self.initial_wavefunction, self.final_wavefunction)
        cmd = [
            "qdtk_propagate.x", "-rst",
            self.initial_wavefunction + ".wfn", "-opr", self.operator + ".op",
            "-gramschmidt", "-dt", str(self.dt), "-tfinal", str(self.tfinal)
        ]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.get_tmp_dir(),
            universal_newlines=True)

        return process

    def _watch_output(self, process):
        re_output = re.compile(
            r"^\s+time:\s+(\d+\.\d+(?:[eE][+-]\d+)?)\s+done$")

        # print live info about run
        try:
            import progressbar
            with progressbar.ProgressBar(
                    max_value=self.tfinal + self.dt) as bar:
                for line in iter(process.stdout.readline, ""):
                    m = re_output.match(line)
                    if not m:
                        continue
                    bar.update(float(m.group(1)))
        except ImportError:
            for line in iter(process.stdout.readline, ""):
                m = re_output.match(line)
                if not m:
                    continue
                logging.info("Finished step to: %f", float(m.group(1)))

    def run(self):
        if self.is_up_to_date():
            logging.info(
                self.type + " task \"%s\" -> \"%s\" is up-to-date, skip",
                self.initial_wavefunction, self.final_wavefunction)
            return

        # prepare temporary dir
        tmp_dir = self.create_tmp_dir()

        # copy initial wave function
        self._copy_initial_wavefunction()

        # copy operator
        self._copy_operator()

        # run program
        process = self._execute()
        self._watch_output(process)
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, cmd)

        logging.info("Finished relaxation procedure: %s -> %s",
                     self.initial_wavefunction, self.final_wavefunction)

        self._create_output_dirs()
        self._symlink_final_wavefunction()
        self._symlink_natpop()
        self._symlink_gpop()

        # write hash
        self.write_hash_file()

    def update_project(self, proj):

        def dummy():
            raise RuntimeError(
                "The wavefunction is created with relaxation task \"%s\"".
                format(self.name))

        proj.add_wavefunction(self.final_wavefunction, dummy)
