from ..inout.projection import read_basis_projection, to_ndarray
from ..tools.wave_function_manipulation import load_wave_function
from .task import Task, FileInput, FileOutput
import os.path
import numpy


class FidelityTask(Task):
    @staticmethod
    def compose_name(wfn, **kwargs):
        return "fidelity_{}".format(wfn).replace("/", "_")

    def __init__(self, wfn, projection, **kwargs):
        kwargs["task_type"] = "FidelityTask"

        self.wfn = wfn
        self.projection = projection

        self.directory = os.path.dirname(self.wfn)
        self.fidelity_name = os.path.join(
            self.directory, "fidelity_{}".format(os.path.basename(wfn)))

        inp_wfn = FileInput("wfn", self.wfn)
        inp_proj = FileInput("projection", self.projection)
        out_fidelity = FileOutput("fidelity", self.fidelity_name)

        Task.__init__(
            self,
            FidelityTask.compose_name(self.wfn),
            self.compute_fidelity,
            inputs=[inp_wfn, inp_proj],
            outputs=[out_fidelity],
            **kwargs)

    def compute_fidelity(self):
        proj = to_ndarray(read_basis_projection(self.projection))[0, 1:]
        coeffs = load_wave_function(self.wfn + ".wfn").PSI[:len(proj)]
        r = numpy.abs(numpy.sum(numpy.conjugate(coeffs) * proj))**2

        with open(self.fidelity_name, "w") as fhout:
            fhout.write(str(r) + "\n")
