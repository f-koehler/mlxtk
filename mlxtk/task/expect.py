from mlxtk import hashing

from distutils.spawn import find_executable
import os


class ExpectationValueTask:
    def __init__(self, project, operator, wavefunction, **kwargs):
        self.project = project
        self.operator = operator
        self.wavefunction = wavefunction

        self.psi = kwargs.get("psi", None)
        self.opsi = kwargs.get("opsi", None)

        self.logger = kwargs.get("logger", project.get_logger("expect"))

    def check_conflicts(self):
        if self.operator not in self.project.operators:
            self.logger.critical("unknown operator \"%s\"", self.operator)
            raise RuntimeError("unknown operator \"{}\"".format(self.operator))

        if self.wavefunction not in self.project.wavefunctions:
            self.logger.critical("unknown wave function \"%s\"", self.initial)
            raise RuntimeError(
                "unknown wave function \"{}\"".format(self.initial))

    def find_exe(self):
        exe = find_executable("qdtk_expect.x")
        if not exe:
            self.logger.critical("cannot find qdtk_expect.x")
            raise RuntimeError("failed to find qdtk_expect.x")
        return exe

    def set_project_targets(self):
        self.check_conflicts()

    def get_task_dir(self):
        return os.path.dirname(
            self.project.wavefunctions[self.wavefunction]["path"])

    def get_output_filename(self):
        if self.psi:
            return "expval_{}".format(self.operator)
        return "expval_{}_{}".format(self.operator, self.wavefunction)

    def get_command_hash_file_path(self):
        return os.path.join(self.get_task_dir() + "_cmd.hash")

    def get_executable_hash_file_path(self):
        return os.path.join(self.get_task_dir(),
                            self.get_output_filename() + "_qdtk_expect.x.hash")

    def is_up_to_date(self):
        self.check_conflicts()

        # check if wave function changed
        if self.project.wavefunctions[self.wavefunction]["updated"]:
            self.logger.info("wave function changed")
            return False

        # check if expectation value exists
        if not os.path.exists(self.get_output_filename()):
            self.logger("expectation value does not exist")
            return False

        # check if command hash file exists
        if not os.path.exists(self.get_command_hash_file_path()):
            self.logger("command hash file does not exist")
            return False

        # check if command hash matches
        with open(self.get_command_hash_file_path()) as fh:
            hash_current = fh.read().strip()

        hash_new = self.get_command_hash()
        if hash_current != hash_new:
            self.logger("parameters changed")
            self.debug("%s != %s", hash_current, hash_new)
            return False

        # check if qdtk_expect.x hash file exists
        if not os.path.exists(self.get_executable_hash_file_path()):
            self.logger("qdtk_expect.x hash file does not exist")
            return False

        # check hash of qdtk_propagate.x
        with open(self.get_executable_hash_file_path()) as fh:
            hash_current = fh.read().strip()

        exe = self.find_exe()
        exe_hash = hashing.hash_file(exe)
        if hash_current != exe_hash:
            self.logger.warning("qdtk_expect.x hash changed, " +
                                "consider restarting manually")
            self.logger.debug("%s != %s", hash_current, exe_hash)

        return True

    def compose_command(self):
        cmd = [
            self.find_exe(), "-rst", self.wavefunction, "-opr", self.operator,
            "-save",
            self.get_output_filename()
        ]
        if self.psi:
            cmd += ["-psi", self.psi]
        if self.opsi:
            cmd += ["-opsi", self.opsi]

        return cmd

    def get_command_hash(self):
        return hashing.hash_string("".join(self.compose_command()))
