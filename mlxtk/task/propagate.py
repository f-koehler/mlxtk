from distutils.spawn import find_executable
from mlxtk import hashing
from mlxtk import log
from mlxtk.inout import output as io_output

import os
import shutil
import subprocess
import threading


class PropagationTask:
    def __init__(self, project, initial, final, hamiltonian, **kwargs):
        self.project = project

        # required arguments
        self.initial = initial
        self.final = final
        self.hamiltonian = hamiltonian

        self.dt = kwargs.get("dt", 0.1)
        self.tfinal = kwargs.get("tfinal", self.dt)

        # operation modes
        self.relax = kwargs.get("relax", False)
        self.improved_relax = kwargs.get("improved_relax", False)
        self.exact_diag = kwargs.get("exact_diag", False)

        # optional arguments
        self.psi = kwargs.get("psi", True)
        self.cont = kwargs.get("cont", False)
        self.rstzero = kwargs.get("rstzero", True)
        self.trans_mat = kwargs.get("trans_mat", False)
        self.mbop_apply = kwargs.get("MBop_apply", False)

        # integrator options
        self.itg = kwargs.get("itg", "zvode")
        self.zvode_mf = kwargs.get("zvode_mf", 10)
        self.atol = kwargs.get("atol", 1e-12)
        self.rtol = kwargs.get("rtol", 1e-11)
        self.reg = kwargs.get("reg", 1e-8)

        # projector / equations of motion
        self.exproj = kwargs.get("exproj", False)
        self.resetnorm = kwargs.get("resetnorm", False)
        self.gramschmidt = kwargs.get("gramschmidt", False)

        # relaxation options
        self.statsteps = kwargs.get("statsteps", 0)
        self.stat_energy_tol = kwargs.get("stat_energy_tol", 1e-8)
        self.stat_npop_tol = kwargs.get("stat_npop_tol", 1e-8)

        # improved relaxation options
        self.nstep_diag = kwargs.get("nstep_diag", 25)
        self.eig_index = kwargs.get("eig_index", 0)

        # exact diagonalization options
        self.eig_tot = kwargs.get("eig_tot", 1)
        self.energy_only = kwargs.get("energy_only", False)

        # create logger
        self.logger = kwargs.get("logger", log.getLogger("propagate"))

        # determine task name and directory
        if self.relax:
            self.task_name = "relax_{}_{}".format(initial, final)
        elif self.improved_relax:
            self.task_name = "improved_relax_{}_{}".format(initial, final)
        elif self.exact_diag:
            self.task_name = "exact_diag_{}".format(initial)
        else:
            self.task_name = "propagate_{}_{}".format(initial, final)
        self.task_dir = os.path.join(self.project.root_dir, self.task_name)

    def is_up_to_date(self):
        # check if initial wave function exists
        if self.initial not in self.project.wavefunctions:
            self.logger.critical("unknown initial wave function \"%s\"",
                                 self.initial)
            raise RuntimeError(
                "unknown wave function \"{}\"".format(self.initial))

        # check if hamiltonian exists
        if self.hamiltonian not in self.project.operators:
            self.logger.critical("unknown operator \"%s\"", self.hamiltonian)
            raise RuntimeError(
                "unknown operator \"{}\"".format(self.hamiltonian))
        self.logger.info("use hamiltonian \"%s\"", self.hamiltonian)

        # check for name conflict
        if self.final in self.project.wavefunctions:
            self.logger.critical("wave function \"%s\" already exists",
                                 self.initial)
            raise RuntimeError(
                "name conflict: wave function \"{}\" already exists".format(
                    self.final))

        # check if initial wave function changed
        if self.project.wavefunctions[self.initial]["updated"]:
            self.logger.info("initial wave function changed")
            return False

        # check if hamiltonian changed
        if self.project.operators[self.hamiltonian]["updated"]:
            self.logger.info("hamiltonian changed")
            return False

        # create task dir if not present
        if not os.path.exists(self.task_dir):
            os.makedirs(self.task_dir)
            self.logger.info("create task directory")
            return False

        # check if qdtk_propagate hash file exists
        exe_hash_file = os.path.join(self.task_dir, "qdtk_propagate.x.hash")
        if not os.path.exists(exe_hash_file):
            self.logger.info("qdtk_propagate.x hash file does not exist")
            return False

        # check hash of qdt_propagate.x
        with open(exe_hash_file) as fh:
            hash_current = fh.read().strip()

        exe = self.find_exe()
        exe_hash = hashing.hash_file(exe)
        if hash_current != exe_hash:
            self.logger.warning("".join([
                "qdtk_propagate.x hash changed,",
                "consider restarting manually"
            ]))
            self.logger.debug("  %s != %s", hash_current, exe_hash)

        # check if last run was complete
        if not self.exact_diag:
            times = io_output.read_output(
                os.path.join(self.task_dir, "output")).time.values
            if times.max() < self.tfinal:
                self.logger.info("last run was incomplete")
                self.cont = True
                return False

        # check if cmd hash file exists
        cmd_hash_file = os.path.join(self.task_dir, "cmd.hash")
        if not os.path.exists(cmd_hash_file):
            self.logger.info("command hash file does not exist")
            return False

        # check parameter hash
        cmd_hash = self.calc_command_hash()
        with open(cmd_hash_file) as fh:
            hash_current = fh.read().strip()

        if hash_current != cmd_hash:
            self.logger.info("propagation parameters changed")
            self.logger.debug("  %s != %s", hash_current, cmd_hash)
            return False

        return True

    def execute(self):
        if self.relax or self.improved_relax:
            self.logger.info("task: relax wave function \"%s\" -> \"%s\"",
                             self.initial, self.final)
        else:
            self.logger.info("task: propagate wave function \"%s\" -> \"%s\"",
                             self.initial, self.final)

        # compose the call to qdtk_propagate
        cmd = self.compose_command()
        self.logger.info("command: %s", " ".join(cmd))

        # check if task is up-to-date
        if self.is_up_to_date():
            self.logger.info("task already up-to-date")
            self.project.wavefunctions[self.final] = {
                "updated": False,
                "path": os.path.join(self.task_dir, "restart")
            }
            return

        # copy initial wave function
        self.logger.info("copy initial wave function to task dir")
        shutil.copy2(self.project.wavefunctions[self.initial]["path"],
                     os.path.join(self.task_dir, self.initial))

        # copy hamiltonian
        self.logger.info("copy hamiltonian to task dir")
        shutil.copy2(self.project.operators[self.hamiltonian]["path"],
                     os.path.join(self.task_dir, self.hamiltonian))

        # run qdtk_propagate.x
        process = subprocess.Popen(
            cmd,
            cwd=self.task_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

        def log_stdout(pipe):
            l = log.getLogger("qdtk_propagate.x")
            with pipe:
                for line in iter(pipe.readline, ""):
                    l.info(line.strip('\n'))

        def log_stderr(pipe):
            l = log.getLogger("qdtk_propagate.x")
            with pipe:
                for line in iter(pipe.readline, ""):
                    l.warning(line.strip('\n'))

        threading.Thread(target=log_stdout, args=[process.stdout]).start()
        threading.Thread(target=log_stderr, args=[process.stderr]).start()

        # wait for process to terminate
        return_code = process.wait()
        if return_code:
            self.logger.critical("qdtk_propagate.x failed with code %d",
                                 return_code)
            raise subprocess.CalledProcessError(return_code, cmd)

        # remove initial wave function
        self.logger.info("remove initial wave function")
        os.remove(os.path.join(self.task_dir, self.initial))

        # remove hamiltonian
        self.logger.info("remove hamiltonian")
        os.remove(os.path.join(self.task_dir, self.hamiltonian))

        # create parameter file
        cmd_hash_file = os.path.join(self.task_dir, "cmd.hash")
        self.logger.info("write cmd hash file")
        with open(cmd_hash_file, "w") as fh:
            fh.write(self.calc_command_hash())

        # create qdtk_propagate hash file
        exe_hash_file = os.path.join(self.task_dir, "qdtk_propagate.x.hash")
        self.logger.info("write qdtk_propagate.x hash file")
        with open(exe_hash_file, "w") as fh:
            fh.write(hashing.hash_file(cmd[0]))

        # mark wave function as updated
        self.project.wavefunctions[self.final] = {
            "updated": False,
            "path": os.path.join(self.task_dir, "restart")
        }
        self.logger.info("done")

    def find_exe(self):
        exe = find_executable("qdtk_propagate.x")
        if not exe:
            self.logger.critical("cannot find qdtk_propagate.x")
            raise RuntimeError("failed to find qdtk_propagate.x")
        return exe

    def compose_command(self):
        cmd = [
            self.find_exe(), "-rst", self.initial, "-opr", self.hamiltonian,
            "-dt",
            str(self.dt), "-tfinal",
            str(self.tfinal)
        ]
        if self.psi:
            cmd.append("-psi")
        if self.relax:
            cmd += [
                "-relax", "-statsteps",
                str(self.statsteps), "-stat_energy_tol",
                str(self.stat_energy_tol), "-stat_npop_tol",
                str(self.stat_npop_tol)
            ]
        if self.improved_relax:
            cmd += [
                "-improved_relax", "-nstep_diag",
                str(self.nstep_diag), "-eig_index",
                str(self.eig_index)
            ]
        if self.exact_diag:
            cmd += ["-exact_diag", "-eig_tot", str(self.eig_tot)]
            if self.energy_only:
                cmd.append("-energy_only")
        if self.cont:
            cmd.append("-cont")
        if self.rstzero:
            cmd.append("-rstzero")
        if self.trans_mat:
            cmd.append("-trans_mat")
        if self.mbop_apply:
            cmd.append("-MBop_apply")
        cmd += ["-itg", self.itg]
        if self.itg == "zvode":
            cmd += ["-zvode_mf", str(self.zvode_mf)]
        cmd += [
            "-atol",
            str(self.atol), "-rtol",
            str(self.rtol), "-reg",
            str(self.reg)
        ]
        if self.exproj:
            cmd.append("-exproj")
        if self.resetnorm:
            cmd.append("-resetnorm")
        if self.gramschmidt:
            cmd.append("-gramschmidt")

        return cmd

    def calc_command_hash(self):
        cmd = self.compose_command()[2:]
        if "-cont" in cmd:
            cmd.remove("-cont")
        return hashing.hash_string("".join(cmd))
