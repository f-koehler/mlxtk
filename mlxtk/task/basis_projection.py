import os
import shutil
import sys
from .task import Task, FileInput, FileOutput
from ..qdtk_executable import find_qdtk_executable
from ..process import watch_process
from ..util import TemporaryCopy


class TDBasisProjectionTask(Task):
    def __init__(self, propagation, basis, **kwargs):
        kwargs["task_type"] = "TDBasisProjectionTask"

        self.propagation = propagation
        self.basis = basis

        self.initial_wave_function = "initial.wave_function"
        self.output_name = "projection_psi_onto_" + basis

        inp_wave_function = FileInput("initial_wave_function",
                                      self.initial_wave_function)
        inp_psi_file = FileInput("psi_file",
                                 os.path.join(propagation.propagation_name,
                                              "psi"))
        inp_basis = FileInput("basis", self.basis)
        out_projection = FileOutput(self.output_name,
                                    os.path.join(propagation.propagation_name,
                                                 self.output_name))

        Task.__init__(
            self,
            "{}_projection_psi_onto_{}".format(
                propagation.propagation_name,
                propagation.initial_wave_function, basis),
            self.compute_projection,
            inputs=[inp_wave_function, inp_psi_file, inp_basis],
            outputs=[out_projection],
            **kwargs)

        if not propagation.psi:
            self.logger.warn("propagation does not seem to create a psi file")

    def get_command(self):
        program_path = find_qdtk_executable("qdtk_analysis.x")
        self.logger.info("use analysis executable: " + program_path)

        return [
            program_path, "-fixed_ns", "-psi", "psi", "-rst_ket",
            self.initial_wave_function, "-rst_bra",
            self.basis + ".wave_function", "-save", self.output_name
        ]

    def compute_projection(self):
        working_dir = self.propagation.propagation_name
        with TemporaryCopy(self.basis + ".wave_function",
                           os.path.join(working_dir,
                                        self.basis + ".wave_function")):
            self.logger.info("run qdtk_analysis.x")
            command = self.get_command()
            self.logger.debug("command: %s", " ".join(command))
            self.logger.debug("working directory: %s", working_dir)

            watch_process(
                command,
                self.logger.info,
                self.logger.warn,
                cwd=working_dir,
                stdout=sys.stdout,
                stderr=sys.stderr)


class BasisProjectionTask(Task):
    def __init__(self, wave_function, basis, **kwargs):
        kwargs["task_type"] = "BasisProjectionTask"

        self.directory = os.path.dirname(wave_function)
        self.wave_function = os.path.basename(wave_function) + ".wave_function"
        self.basis = basis
        self.output_name = "projection_" + os.path.splitext(self.wave_function)[0] + "_onto_" + basis

        inp_wave_function = FileInput("wave_function", self.wave_function)
        inp_basis = FileInput("basis", self.basis)
        out_projection = FileOutput(self.output_name,
                                    os.path.join(self.directory,
                                                 self.output_name))

        Task.__init__(
            self,
            "{}_projection_{}_onto_{}".format(self.directory,
                                              self.wave_function, self.basis),
            self.compute_projection,
            inputs=[inp_wave_function, inp_basis],
            outputs=[out_projection],
            **kwargs)

    def get_command(self):
        program_path = find_qdtk_executable("qdtk_analysis.x")
        self.logger.info("use analysis executable: " + program_path)

        return [
            program_path, "-fixed_ns", "-rst_ket", self.wave_function,
            "-rst_bra", self.basis + ".wave_function", "-save",
            self.output_name
        ]

    def compute_projection(self):
        with TemporaryCopy(self.basis + ".wave_function",
                           os.path.join(self.directory,
                                        self.basis + ".wave_function")):
            self.logger.info("run qdtk_analysis.x")
            command = self.get_command()
            self.logger.debug("command: %s", " ".join(command))
            self.logger.debug("working directory: %s", self.directory)

            watch_process(
                command,
                self.logger.info,
                self.logger.warn,
                cwd=self.directory,
                stdout=sys.stdout,
                stderr=sys.stderr)
