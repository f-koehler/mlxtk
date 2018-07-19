import h5py
import os
import sys

from ..inout import hdf5
from ..process import watch_process
from ..qdtk_executable import find_qdtk_executable
from ..util import TemporaryCopy
from .task import Task, FileInput, FileOutput
from ..inout.projection import add_projection_to_hdf5


class BasisProjectionTask(Task):
    @staticmethod
    def compose_name(wave_function, basis):
        return "basis_projection_{}_onto_{}".format(wave_function,
                                                    basis).replace("/", "_")

    def __init__(self, wave_function, basis, **kwargs):
        kwargs["task_type"] = "BasisProjectionTask"

        self.directory = os.path.dirname(wave_function)
        self.wave_function = wave_function + ".wfn"
        self.basis = basis

        self.basename = ("basis_projection_" + os.path.splitext(
            os.path.basename(self.wave_function))[0] + "_onto_" +
                         os.path.relpath(basis, self.directory))
        self.projection_name = os.path.join(self.directory,
                                            self.basename.replace("/", "_"))

        inp_wave_function = FileInput("wave_function", self.wave_function)
        inp_basis = FileInput("basis", self.basis)
        out_projection = FileOutput(
            self.projection_name, os.path.join(self.directory, self.basename))

        Task.__init__(
            self,
            BasisProjectionTask.compose_name(wave_function, basis),
            self.compute_projection,
            inputs=[inp_wave_function, inp_basis],
            outputs=[out_projection],
            **kwargs)

    def get_command(self):
        program_path = find_qdtk_executable("qdtk_analysis.x")
        self.logger.info("use analysis executable: " + program_path)

        return [
            program_path,
            "-fixed_ns",
            "-rst_ket",
            os.path.relpath(self.wave_function, self.directory),
            "-rst_bra",
            os.path.relpath(self.basis + ".wfn", self.directory),
            "-save",
            os.path.relpath(self.projection_name, self.directory),
        ]

    def compute_projection(self):
        with TemporaryCopy(
                self.basis + ".wfn",
                os.path.join(self.basis + ".wfn"),
        ):
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
                stderr=sys.stderr,
            )

    def create_hdf5(self, group=None):
        if self.is_running():
            raise hdf5.IncompleteHDF5(
                "Possibly running BasisProjectionTask, no data can be added")

        opened_file = group is None
        if opened_file:
            self.logger.info("create new HDF5 file")
            group = h5py.File(self.projection_name + ".hdf5", "w")

        path = os.path.join(self.directory, self.projection_name)

        if not os.path.exists(path):
            raise FileNotFoundError(
                "BasisProjection file \"{}\" does not exist".format(path))

        add_projection_to_hdf5(group, path, self.projection_name)

        if opened_file:
            group.close()
