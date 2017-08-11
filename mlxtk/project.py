from mlxtk import hash
from mlxtk import log
from mlxtk.inout import output as io_output
from mlxtk.plot import energy as plt_energy
from mlxtk.plot import norm as plt_norm
from QDTK.Operator import Operator
from QDTK.Operatorb import Operatorb

from distutils.spawn import find_executable
import os
import shutil
import subprocess
import sys
import threading
if sys.version_info <= (3, 0):
    from StringIO import StringIO
else:
    from io import StringIO


def create_operator_table(*args):
    return "\n".join(args)


class Project:
    def __init__(self, name, **kwargs):
        self.name = name
        self.root_dir = kwargs.get("root_dir", os.path.join(os.getcwd(), name))
        self.operators = {}
        self.wavefunctions = {}

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

    def create_operator(self, name, func):
        logger = log.getLogger("operator")
        logger.info("task: create operator \"%s\"", name)

        # compute the path of the operator file
        output_path = os.path.join(self.root_dir, name + ".opr")

        # check for name conflict
        if name in self.operators:
            logger.critical("operator \"%s\" already exists", name)
            raise RuntimeError(
                "name conflict: operator \"{}\" already exists".format(name))

        # calculate the operator and get the contents for the new operator file
        sio = StringIO()
        operator = func()
        if isinstance(operator, Operator):
            operator.createOperatorFile(sio)
        else:
            operator.createOperatorFileb(sio)
        string = sio.getvalue()

        # create an entry in the operator dictionary
        self.operators[name] = {"path": output_path}

        # check if operator already up-to-date
        if os.path.exists(output_path):
            hash_current = hash.hash_file(output_path)
            hash_new = hash.hash_string(string)
            if hash_current == hash_new:
                logger.info("already up-to-date")
                self.operators[name]["updated"] = False
                return
            else:
                logger.info("hash mismatch")
                logger.debug("  %s != %s", hash_current, hash_new)
        else:
            logger.info("file does not exist")

        # create the operator file
        logger.info("write operator file")
        with open(output_path, "w") as fh:
            fh.write(string)
        logger.info("done")

        # mark operator update
        self.operators[name]["updated"] = False

    def create_wavefunction(self, name, func):
        logger = log.getLogger("wavefunction")
        logger.info("task: create wave function \"%s\"", name)

        # compute the path of the wave function file
        output_path = os.path.join(self.root_dir, name + ".wfn")

        # check for name conflict
        if name in self.wavefunctions:
            logger.critical("wave function \"%s\" already exists", name)
            raise RuntimeError(
                "name conflict: wave function \"{}\" already exists".format(
                    name))

        # compute wave function and get the contents of the new file
        sio = StringIO()
        func().createWfnFile(sio)
        string = sio.getvalue()

        # create an entry in the wave function dictionary
        self.wavefunctions[name] = {"path": output_path}

        # check if wave function already up-to-date
        if os.path.exists(output_path):
            hash_current = hash.hash_file(output_path)
            hash_new = hash.hash_string(string)
            if hash_current == hash_new:
                logger.info("already up-to-date")
                self.wavefunctions[name]["updated"] = False
                return
            else:
                logger.info("hash mismatch")
                logger.debug("  %s != %s", hash_current, hash_new)
        else:
            logger.info("file does not exist")

        # create wave function file
        logger.info("write wave function file")
        with open(output_path, "w") as fh:
            fh.write(string)
        logger.info("done")

        # mark wave function update
        self.wavefunctions[name]["updated"] = True

    def propagate(self, initial, final, hamiltonian, **kwargs):
        # required arguments
        dt = kwargs.get("dt", 0.1)
        tfinal = kwargs.get("tfinal", dt)

        # operation modes
        relax = kwargs.get("relax", False)
        improved_relax = kwargs.get("improved_relax", False)
        exact_diag = kwargs.get("exact_diag", False)

        # optional arguments
        psi = kwargs.get("psi", True)
        cont = kwargs.get("cont", False)
        rstzero = kwargs.get("rstzero", True)
        trans_mat = kwargs.get("trans_mat", False)
        mbop_apply = kwargs.get("MBop_apply", False)

        # integrator options
        itg = kwargs.get("itg", "zvode")
        zvode_mf = kwargs.get("zvode_mf", 10)
        atol = kwargs.get("atol", 1e-12)
        rtol = kwargs.get("rtol", 1e-11)
        reg = kwargs.get("reg", 1e-8)

        # projector / equations of motion
        exproj = kwargs.get("exproj", False)
        resetnorm = kwargs.get("resetnorm", False)
        gramschmidt = kwargs.get("gramschmidt", False)

        # relaxation options
        statsteps = kwargs.get("statsteps", 0)
        stat_energy_tol = kwargs.get("stat_energy_tol", 1e-8)
        stat_npop_tol = kwargs.get("stat_npop_tol", 1e-8)

        # improved relaxation options
        nstep_diag = kwargs.get("nstep_diag", 25)
        eig_index = kwargs.get("eig_index", 0)

        # exact diagonalization options
        eig_tot = kwargs.get("eig_tot", 1)
        energy_only = kwargs.get("energy_only", False)

        up_to_date = True

        # create logger
        logger = kwargs.get("logger", log.getLogger("propagate"))
        if relax or improved_relax:
            logger.info("task: relax wave function \"%s\" -> \"%s\"", initial,
                        final)
        else:
            logger.info("task: propagate wave function \"%s\" -> \"%s\"",
                        initial, final)

        # check if initial wave function exists
        if initial not in self.wavefunctions:
            logger.critical("unknown initial wave function \"%s\"", initial)
            raise RuntimeError("unknown wave function \"{}\"".format(initial))

        # check if initial wave function changed
        if self.wavefunctions[initial]["updated"]:
            logger.info("initial wave function changed")
            up_to_date = False

        # check for name conflict
        if final in self.wavefunctions:
            logger.critical("wave function \"%s\" already exists", initial)
            raise RuntimeError(
                "name conflict: wave function \"{}\" already exists".format(
                    final))

        # check if hamiltonian exists
        if hamiltonian not in self.operators:
            logger.critical("unknown operator \"%s\"", hamiltonian)
            raise RuntimeError("unknown operator \"{}\"".format(hamiltonian))
        logger.info("use hamiltonian \"%s\"", hamiltonian)

        # check if hamiltonian changed
        if self.operators[hamiltonian]["updated"]:
            if up_to_date:
                logger.info("hamiltonian changed")
                up_to_date = False

        # determine task name and directory
        if relax:
            task_name = "relax_{}_{}".format(initial, final)
        elif improved_relax:
            task_name = "improved_relax_{}_{}".format(initial, final)
        elif exact_diag:
            task_name = "exact_diag_{}".format(initial)
        else:
            task_name = "propagate_{}_{}".format(initial, final)
        task_dir = os.path.join(self.root_dir, task_name)

        # create task dir if not present
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)
            if up_to_date:
                logger.info("create task directory")
                up_to_date = False

        # find qdtk_propagate.x and compute hash
        qdtk_executable = find_executable("qdtk_propagate.x")
        if not qdtk_executable:
            logger.critical("cannot find qdtk_propagate.x")
            raise RuntimeError("failed to find qdtk_propagate.x")
        qdtk_executable_hash = hash.hash_file(qdtk_executable)

        # check if qdtk_propagate hash file exists
        qdtk_executable_hash_file = os.path.join(task_dir,
                                                 "qdtk_propagate.hash")
        if not os.path.exists(qdtk_executable_hash_file) and up_to_date:
            logger.info("qdtk_propagate hash file does not exist")
            up_to_date = False

        # check hash of qdt_propagate.x
        if up_to_date:
            with open(qdtk_executable_hash_file) as fh:
                hash_current = fh.read().strip()

            if hash_current != qdtk_executable_hash:
                logger.warning("".join([
                    "qdtk_propagate.x hash changed,",
                    "consider restarting manually"
                ]))
                logger.debug("  %s != %s", hash_current, qdtk_executable_hash)

        # check if last run was complete
        if up_to_date:
            times = io_output.read_output(
                os.path.join(task_dir, "output")).time.values
            if times.max() < tfinal:
                logger.info("last run was incomplete")
                up_to_date = False
                cont = True

        # copy initial wave function
        logger.info("copy initial wave function to task dir")
        shutil.copy2(self.wavefunctions[initial]["path"],
                     os.path.join(task_dir, initial))

        # copy hamiltonian
        logger.info("copy hamiltonian to task dir")
        shutil.copy2(self.operators[hamiltonian]["path"],
                     os.path.join(task_dir, hamiltonian))

        # compose the call to qdtk_propagate
        cmd = [
            qdtk_executable, "-rst", initial, "-opr", hamiltonian, "-dt",
            str(dt), "-tfinal",
            str(tfinal)
        ]
        if psi:
            cmd.append("-psi")
        if relax:
            cmd += [
                "-relax", "-statsteps",
                str(statsteps), "-stat_energy_tol",
                str(stat_energy_tol), "-stat_npop_tol",
                str(stat_npop_tol)
            ]
        if improved_relax:
            cmd += [
                "-improved_relax", "-nstep_diag",
                str(nstep_diag), "-eig_index",
                str(eig_index)
            ]
        if exact_diag:
            cmd += ["-exact_diag", "-eig_tot", str(eig_tot)]
            if energy_only:
                cmd.append("-energy_only")
        if cont:
            cmd.append("-cont")
        if rstzero:
            cmd.append("-rstzero")
        if trans_mat:
            cmd.append("-trans_mat")
        if mbop_apply:
            cmd.append("-MBop_apply")
        cmd += ["-itg", itg]
        if itg == "zvode":
            cmd += ["-zvode_mf", str(zvode_mf)]
        cmd += ["-atol", str(atol), "-rtol", str(rtol), "-reg", str(reg)]
        if exproj:
            cmd.append("-exproj")
        if resetnorm:
            cmd.append("-resetnorm")
        if gramschmidt:
            cmd.append("-gramschmidt")
        logger.info("command: %s", " ".join(cmd))

        # check if cmd hash file exists
        cmd_hash_file = os.path.join(task_dir, "cmd.hash")
        if not os.path.exists(cmd_hash_file) and up_to_date:
            logger.info("command hash file does not exist")
            up_to_date = False

        # check parameter hash
        cmd_hash = hash.hash_string("".join(cmd[1:]))
        if up_to_date:
            with open(cmd_hash_file) as fh:
                hash_current = fh.read().strip()

            if hash_current != cmd_hash:
                logger.info("propagation parameters changed")
                logger.debug("  %s != %s", hash_current, cmd_hash)
                up_to_date = False

        # check if task is up-to-date
        if up_to_date:
            logger.info("task already up-to-date")
            self.wavefunctions[final] = {
                "updated": False,
                "path": os.path.join(task_dir, "restart")
            }
            return

        process = subprocess.Popen(
            cmd, cwd=task_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

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
            logger.critical("qdtk_propagate.x failed with code %d",
                            return_code)
            raise subprocess.CalledProcessError(return_code, cmd)

        # remove initial wave function
        logger.info("remove initial wave function")
        os.remove(os.path.join(task_dir, initial))

        # remove hamiltonian
        logger.info("remove hamiltonian")
        os.remove(os.path.join(task_dir, hamiltonian))

        # create parameter file
        logger.info("write cmd hash file")
        with open(cmd_hash_file, "w") as fh:
            fh.write(cmd_hash)

        # create qdtk_propagate hash file
        logger.info("write qdtk_propagate.x hash file")
        with open(qdtk_executable_hash_file, "w") as fh:
            fh.write(qdtk_executable_hash)

        # mark wave function as updated
        self.wavefunctions[final] = {
            "updated": False,
            "path": os.path.join(task_dir, "restart")
        }
        logger.info("done")

        if not exact_diag:
            self.plot_output(os.path.join(task_dir, "output"))

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

    def plot_output(self, path):
        logger = log.getLogger("plot")

        logger.info("reading output data from \"%s\"", path)
        data = io_output.read_output(path)

        logger.info("plotting energy")
        plt_energy.plot_energy(data).save(
            os.path.join(os.path.dirname(path), "energy.pdf"))

        logger.info("plotting norm")
        plt_norm.plot_norm(data).save(
            os.path.join(os.path.dirname(path), "norm.pdf"))
