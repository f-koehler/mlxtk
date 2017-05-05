from mlxtk.task.task import Task
from mlxtk.filesystem import relative_symlink
import mlxtk.log as log

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
        self.reset_time = kwargs.get("reset_time", True)

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
            log.debug("Task %s not up-to-date, hash file does not exist",
                      self.name)
            return False

        if self.hash() != self.read_hash_file():
            log.debug("Task %s not up-to-date, hash values differ", self.name)
            return False

        if not os.path.exists(output_path):
            log.debug(
                "Task %s not up-to-date, final wave function does not exist",
                self.name)
            return False

        if os.path.getmtime(input_path) > os.path.getmtime(output_path):
            log.debug(
                "Task %s not up-to-date, initial wave function is newer than the resulting one",
                self.name)
            return False

        if os.path.getmtime(operator_path) > os.path.getmtime(output_path):
            log.debug(
                "Task %s not up-to-date, Hamiltonian is newer than the resulting wave function",
                self.name)
            return False

        log.debug("Task %s is up-to-date", self.name)
        return True

    def get_input_path(self):
        return os.path.join("wavefunctions", self.initial_wavefunction + ".wfn")

    def get_output_path(self):
        return os.path.join("wavefunctions", self.final_wavefunction + ".wfn")

    def get_operator_path(self):
        return os.path.join("operators", self.operator + ".op")

    def _copy_operator(self):
        dst = os.path.join(self.get_working_dir(), self.operator + ".op")
        log.info("Copy Hamiltonian: %s -> %s", self.get_operator_path(), dst)
        shutil.copy(self.get_operator_path(), dst)

    def _copy_initial_wavefunction(self):
        dst = os.path.join(self.get_working_dir(),
                           self.initial_wavefunction + ".wfn")
        log.info("Copy initial wave function: %s -> %s",
                 self.get_input_path(), dst)
        shutil.copy(self.get_input_path(), dst)

    def _symlink_final_wavefunction(self):
        src = os.path.join(self.get_working_dir(), "restart")
        dst = self.get_output_path()
        log.info("Symlink initial wave function: %s -> %s", src, dst)
        relative_symlink(src, dst)

    def _symlink_natpop(self):
        src = os.path.join(self.get_working_dir(), "natpop")
        dst = os.path.join("natural_populations", self.name + ".natpop")
        log.info("Symlink natural population: %s -> %s", src, dst)
        relative_symlink(src, dst)

    def _symlink_gpop(self):
        src = os.path.join(self.get_working_dir(), "gpop")
        dst = os.path.join("gpops", self.name + ".gpop")
        log.info("Symlink gpop: %s -> %s", src, dst)
        relative_symlink(src, dst)

    def _symlink_output(self):
        src = os.path.join(self.get_working_dir(), "output")
        dst = os.path.join("outputs", self.name + ".out")
        log.info("Symlink output: %s -> %s", src, dst)
        relative_symlink(src, dst)

    def _create_output_dirs(self):
        dir = os.path.join("natural_populations")
        if not os.path.exists(dir):
            log.info("Create directory: %s", dir)
            os.mkdir(dir)

        dir = os.path.join("gpops")
        if not os.path.exists(dir):
            log.info("Create directory: %s", dir)
            os.mkdir(dir)

        dir = os.path.join("outputs")
        if not os.path.exists(dir):
            log.info("Create directory: %s", dir)
            os.mkdir(dir)

    def _execute(self):
        log.info("Execute propagation procedure: %s -> %s",
                 self.initial_wavefunction, self.final_wavefunction)
        cmd = [
            "qdtk_propagate.x", "-rst", self.initial_wavefunction + ".wfn",
            "-opr", self.operator + ".op", "-gramschmidt", "-dt", str(self.dt),
            "-tfinal", str(self.tfinal)
        ]
        if self.reset_time:
            cmd.append("-rstzero")

        log.debug("Full command: %s", " ".join(cmd))
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.get_working_dir(),
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
                log.info("Finished step to: %f", float(m.group(1)))

    def run(self):
        print("")
        log.draw_box("TASK: {}".format(self.name))
        if self.is_up_to_date():
            log.info(self.type + " task \"%s\" -> \"%s\" is up-to-date, skip",
                     self.initial_wavefunction, self.final_wavefunction)
            return

        # prepare temporary dir
        working_dir = self.create_working_dir()

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

        log.info("Finished relaxation procedure: %s -> %s",
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
