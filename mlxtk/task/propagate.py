from mlxtk import hashing
from mlxtk.inout import output as io_output
from mlxtk.plot.energy import plot_energy
from mlxtk.plot.norm import plot_norm
from mlxtk.plot.overlap import plot_overlap
from mlxtk.process import watch_process

from distutils.spawn import find_executable

import os
import shutil


class PropagationTask:
    """Create a PropagationTask

    Attributes:
        project: Project this task is part of

        initial (str): Name of the initial wave function
        final (str): Name of the final wave function
        hamiltonian (str): Name of the Hamiltonian used for the propagation

        dt (float): Time step size for the integrator
        tfinal (float): Stopping time for the integrator

        relax (bool): Whether this is a relaxation task
        improved_relax (bool): Whether this is an improved relaxation task
        exact_diag (bool): Whether this is an exact diagonalization task

        psi (bool): Whether to create a psi file (containing the wave function after each time step)
        cont (bool): Whether this a continuation of a previous run
        rstzero (bool): Whether to reset the time to zero in the output files
        trans_mat (bool): Whether to write out the transition matrices

    Args:
        project: The project this task is part of
        initial (str): Name of the initial wave function
        final (str): Name of the final wave function
        hamiltonian (str): Name of the Hamilton operator to be used for propagation
    """
    def __init__(self, project, initial, final, hamiltonian, **kwargs):
        self.project = project

        # required arguments
        self.initial = initial
        self.final = final
        self.hamiltonian = hamiltonian

        self.dt = kwargs.get("dt", 1.0)
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

        self.threads = kwargs.get("threads", 1)

        # create logger
        self.logger = kwargs.get("logger",
                                 self.project.get_logger("propagate"))

    def execute(self):
        if self.relax or self.improved_relax:
            self.logger.info("task: relax wave function \"%s\" -> \"%s\"",
                             self.initial, self.final)
        else:
            self.logger.info("task: propagate wave function \"%s\" -> \"%s\"",
                             self.initial, self.final)

        # compose the call to qdtk_propagate
        cmd = self._compose_command()

        if not os.path.exists(self._get_task_dir()):
            self.logger.info("create task directory")
            os.makedirs(self._get_task_dir())

        # check if task is up-to-date
        if self.is_up_to_date():
            self._update_project(False)
            self.logger.info("task already up-to-date")
            self.logger.info("done")
            return

        # copy initial wave function
        self._copy_wave_function()

        # copy hamiltonian
        self._copy_hamiltonian()

        # run qdtk_propagate.x
        self.logger.info("command: %s", " ".join(cmd))
        watch_process(
            cmd,
            self.logger.info,
            self.logger.warning,
            cwd=self._get_task_dir())

        # remove initial wave function
        self._remove_wave_function()

        # remove hamiltonian
        self._remove_hamiltonian()

        # create parameter file
        cmd_hash_file = self._get_command_hash_path()
        self.logger.info("write cmd hash file")
        with open(cmd_hash_file, "w") as fh:
            fh.write(self._get_command_hash())

        # create qdtk_propagate hash file
        exe_hash_file = self._get_exe_hash_path()
        self.logger.info("write qdtk_propagate.x hash file")
        with open(exe_hash_file, "w") as fh:
            fh.write(hashing.hash_file(cmd[0]))

        # mark wave function as updated
        self._update_project(True)

        self.logger.info("done")

    def is_up_to_date(self):
        """Check if the task is up-to-date or not

        Returns:
            bool: True if the project is up-to-date, False otherwise
        """
        self._check_conflicts()

        # create task dir if not present
        if not os.path.exists(self._get_task_dir()):
            self.logger.info("task dir does not exist")
            return False

        # check if task was run at all
        if not os.path.exists(os.path.join(self._get_task_dir(), "output")):
            self.logger.info("output file does not exist")
            return False

        # check if initial wave function changed
        if self._has_initial_wave_function_changed():
            return False

        # check if hamiltonian changed
        if self._has_hamiltonian_changed():
            return False

        # check if last run was complete
        if not self._is_last_run_complete():
            return False

        return self._is_command_hash_valid()

    def plot(self):
        """Plot the results of the propagation
        """
        def get_path(relative):
            return os.path.join(self._get_task_dir(), relative)

        if self.exact_diag:
            return

        self.logger.info("read output file")
        output = io_output.read_output(get_path("output"))

        self.logger.info("plotting energy")
        plot_energy(output).save(get_path("energy.pdf"))

        self.logger.info("plotting norm")
        plot_norm(output).save(get_path("norm.pdf"))

        self.logger.info("plotting overlap")
        plot_overlap(output).save(get_path("overlap.pdf"))

    def set_project_targets(self):
        self._check_conflicts()
        self.project.wavefunctions[self.final] = {
            "path": os.path.join(self._get_task_dir(), "restart")
        }
        if self.psi:
            self.project.psis["{}_{}".format(self.initial, self.final)] = {
                "path": os.path.join(self._get_task_dir(), "psi")
            }

    def _check_conflicts(self):
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

    def _compose_command(self):
        cmd = [
            self._find_executable(), "-rst", self.initial, "-opr",
            self.hamiltonian, "-dt",
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

        if self.threads:
            cmd += ["-threads", str(self.threads)]

        return cmd

    def _copy_hamiltonian(self):
        self.logger.info("copy hamiltonian to task dir")
        shutil.copy2(self.project.operators[self.hamiltonian]["path"],
                     os.path.join(self._get_task_dir(), self.hamiltonian))

    def _copy_wave_function(self):
        self.logger.info("copy initial wave function to task dir")
        shutil.copy2(self.project.wavefunctions[self.initial]["path"],
                     os.path.join(self._get_task_dir(), self.initial))

    def _find_executable(self):
        exe = find_executable("qdtk_propagate.x")
        if not exe:
            self.logger.critical("cannot find qdtk_propagate.x")
            raise RuntimeError("failed to find qdtk_propagate.x")
        return exe

    def _get_command_hash(self):
        cmd = self._compose_command()[2:]
        if "-cont" in cmd:
            cmd.remove("-cont")
        return hashing.hash_string("".join(cmd))

    def _get_command_hash_path(self):
        return os.path.join(
            self.project.get_hash_dir(),
            "propagation_" + self._get_task_name() + "_command")

    def _get_exe_hash_path(self):
        return os.path.join(
            self.project.get_hash_dir(),
            "propagation_" + self._get_task_name() + "_qdtk_propagate.x")

    def _get_task_name(self):
        if self.relax:
            return "relax_{}_{}".format(self.initial, self.final)
        if self.improved_relax:
            return "improved_relax_{}_{}".format(self.initial, self.final)
        if self.exact_diag:
            return "exact_diag_{}".format(self.initial)
        return "propagate_{}_{}".format(self.initial, self.final)

    def _get_task_dir(self):
        return os.path.join(self.project.root_dir, self._get_task_name())

    def _has_hamiltonian_changed(self):
        ret = self.project.operators[self.hamiltonian]["updated"]
        if ret:
            self.logger.info("hamiltonian changed")
        return ret

    def _has_initial_wave_function_changed(self):
        ret = self.project.wavefunctions[self.initial]["updated"]
        if ret:
            self.logger.info("initial wave function changed")
        return ret

    def _is_command_hash_valid(self):
        cmd_hash_file = self._get_command_hash_path()
        if not os.path.exists(cmd_hash_file):
            self.logger.info("command hash file does not exist")
            return False

        # check cmd hash
        cmd_hash = self._get_command_hash()
        with open(self._get_command_hash_path()) as fh:
            hash_current = fh.read().strip()

        if hash_current != cmd_hash:
            self.logger.info("propagation parameters changed")
            self.logger.debug("%s != %s", hash_current, cmd_hash)
            return False
        return True

    def _is_exe_hash_valid(self):
        exe_hash_file = self._get_exe_hash_path()
        if not os.path.exists(exe_hash_file):
            self.logger.info("qdtk_propagate.x hash file does not exist")
            return False

        with open(exe_hash_file) as fh:
            hash_current = fh.read().strip()

        exe = self._find_executable()
        exe_hash = hashing.hash_file(exe)
        if hash_current != exe_hash:
            self.logger.warning("qdtk_propagate.x hash changed, " +
                                "consider restarting manually")
            self.logger.debug("%s != %s", hash_current, exe_hash)
            return False
        return True

    def _is_last_run_complete(self):
        if self.exact_diag:
            return True

        times = io_output.read_output(
            os.path.join(self._get_task_dir(), "output")).time.values
        if times.max() < self.tfinal:
            self.logger.info("last run was incomplete")
            self.cont = True
            return False
        return True

    def _remove_hamiltonian(self):
        self.logger.info("remove hamiltonian")
        os.remove(os.path.join(self._get_task_dir(), self.hamiltonian))

    def _remove_wave_function(self):
        self.logger.info("remove initial wave function")
        os.remove(os.path.join(self._get_task_dir(), self.initial))

    def _update_project(self, updated):
        self.project.wavefunctions[self.final] = {
            "updated": updated,
            "path": os.path.join(self._get_task_dir(), "restart")
        }
        if self.psi:
            self.project.psis["{}_{}".format(self.initial, self.final)] = {
                "updated": updated,
                "path": os.path.join(self._get_task_dir(), "psi")
            }
