import os
import numpy

from QDTK.Wavefunction import Wavefunction

from mlxtk.task import task


class AddMomentum(task.Task):
    def __init__(self, wave_function, new_wave_function, momentum, **kwargs):
        self.wave_function = wave_function
        self.new_wave_function = new_wave_function
        self.momentum = momentum

        kwargs["task_type"] = "AddMomentum"

        task.Task.__init__(
            self,
            "add_momentum" + wave_function,
            None,
            inputs=[],
            outputs=[],
            **kwargs)

        def add_momentum(self):
            wfn = Wavefunction()
            wfn.load(
                os.path.join(self.cwd, self.wave_function + ".wave_function"))

            num_spfs = wfn.tree._subnodes[0]._dim
            len_spfs = wfn.tree._subnodes[0]._phiLen
            grid = wfn.tree._topNode._pgrid[0]

            phase = numpy.exp(-momentum * grid)

            for i in range(0, num_spfs):
                start = wfn.tree._subnodes[0].z0 + i * len_spfs
                stop = start + len_spfs
                wfn.PSI[start:stop] = phase * wfn.PSI[start:stop]
