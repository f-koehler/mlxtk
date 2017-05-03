from mlxtk.task.task import Task

import logging
import os.path
import re
import shutil
import subprocess


class Relaxation(Task):
    def __init__(self, initial_wavefunction, final_wavefunction, operator,
                 **kwargs):
        Task.__init__(self, "relax_{}_to_{}".format(initial_wavefunction,
                                                    final_wavefunction))
        self.initial_wavefunction = initial_wavefunction
        self.final_wavefunction = final_wavefunction
        self.operator = operator

        self.statsteps = kwargs.get("statsteps", 0)
        self.dt = kwargs.get("dt", 1.)
        self.tfinal = kwargs.get("tfinal", self.dt)

    def is_up_to_date(self):
        hash_path = self.get_hash_path()
        input_path = self.get_input_path()
        output_path = self.get_output_path()
        operator_path = self.get_operator_path()

        if not os.path.exists(input_path):
            raise RuntimeError("Wavefunction \"{}\" does not exist (\"{}\")".
                               format(initial_wavefunction, input_path))

        if not os.path.exists(operator_path):
            raise RuntimeError("Operator \"{}\" does not exist (\"{}\")".
                               format(operator, operator_path))

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
        return os.path.join(self.root_dir, "wavefunctions",
                            self.initial_wavefunction + ".wfn")

    def get_output_path(self):
        return os.path.join(self.root_dir, "wavefunctions",
                            self.final_wavefunction + ".wfn")

    def get_operator_path(self):
        return os.path.join(self.root_dir, "operators", self.operator + ".op")

    def run(self):
        if self.is_up_to_date():
            logging.info(
                "Relaxation task \"%s\" -> \"%s\" is up-to-date, skip",
                self.initial_wavefunction, self.final_wavefunction)
            return

        # prepare temporary dir
        tmp_dir = self.create_tmp_dir()

        # copy initial wave function
        tmp_input = os.path.join(tmp_dir, self.initial_wavefunction + ".wfn")
        logging.info("Copy initial wave function: %s -> %s",
                     self.get_input_path(), tmp_input)
        shutil.copy(self.get_input_path(), tmp_input)

        # copy operator
        tmp_operator = os.path.join(tmp_dir, self.operator + ".op")
        logging.info("Copy Hamiltonian: %s -> %s",
                     self.get_operator_path(), tmp_operator)
        shutil.copy(self.get_operator_path(), tmp_operator)

        # run program
        logging.info("Execute relaxation procedure: %s -> %s",
                     self.initial_wavefunction, self.final_wavefunction)
        cmd = [
            "qdtk_propagate.x", "-relax", "-rst",
            self.initial_wavefunction + ".wfn", "-opr", self.operator + ".op",
            "-gramschmidt", "-dt", str(self.dt), "-tfinal", str(self.tfinal),
            "-stateps", str(self.statsteps)
        ]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=tmp_dir,
            universal_newlines=True)

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
                logging.info("Finished step: %f", float(m.group(1)))
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, cmd)

        logging.info("Finished relaxation procedure: %s -> %s",
                     self.initial_wavefunction, self.final_wavefunction)

        # copy resulting wave_function
        tmp_output = os.path.join(tmp_dir, "restart")
        logging.info("Copy initial wave function: %s -> %s", tmp_output,
                     self.get_output_path())
        shutil.copy(tmp_output, self.get_output_path())

        # clean up
        self.clean()

        # write hash
        self.write_hash_file()

    def update_project(self, proj):
        def dummy():
            raise RuntimeError(
                "The wavefunction is created with relaxation task \"%s\"".
                format(self.name))

        proj.add_wavefunction(self.final_wavefunction, dummy)
